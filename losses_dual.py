# losses_dual.py
import torch
import torch.nn.functional as F
from bezier_utils import sample_bezier


def curve_loss(pred_ctrl, gt_polyline, lane_mask=None, num_samples=40, reduction='mean'):
    """
    pred_ctrl: (N, L, 4, 2)
    gt_polyline: (N, L, T, 2) - T == num_samples 라고 가정
    lane_mask: (N, L) or None
    """
    pred_pts = sample_bezier(pred_ctrl, num_samples=num_samples)  # (N, L, T, 2)

    if pred_pts.shape != gt_polyline.shape:
        raise ValueError(
            f"[curve_loss] pred {pred_pts.shape}, gt {gt_polyline.shape} mismatch. "
            f"gt를 num_samples={num_samples}에 맞게 resample 해야 한다."
        )

    diff = torch.abs(pred_pts - gt_polyline)  # (N, L, T, 2)
    if lane_mask is not None:
        lane_mask = lane_mask.view(pred_pts.shape[0], pred_pts.shape[1], 1, 1)
        diff = diff * lane_mask

    loss = diff.mean(dim=(-1, -2))  # (N, L)
    if reduction == 'mean':
        loss = loss.mean()
    return loss


def straight_loss(pred_straight, gt_anchor, lane_mask=None, reduction='mean'):
    """
    pred_straight:
      {
        'x': (B,L,R),
        'exist_logit': (B,L,R),
      }
    gt_anchor:
      {
        'x': (B,L,R),
        'mask': (B,L,R),
        'exist': (B,L,),
      }
    """
    pred_x = pred_straight['x']              # (B,L,R)
    gt_x = gt_anchor['x']                    # (B,L,R)
    mask = gt_anchor['mask']                 # (B,L,R)

    # x 회귀 L1
    diff = torch.abs(pred_x - gt_x) * mask
    loss_x = diff.sum() / (mask.sum() + 1e-6)

    # 존재 여부 BCE
    exist_logit = pred_straight['exist_logit']        # (B,L,R)
    gt_exist = (gt_anchor['mask'] > 0.5).float()      # (B,L,R)

    bce = F.binary_cross_entropy_with_logits(
        exist_logit, gt_exist, reduction='none'
    )
    loss_exist = (bce * mask).sum() / (mask.sum() + 1e-6)

    # 가중치는 필요에 따라 조정 가능
    return loss_x + 0.1 * loss_exist


def consistency_loss(outputs, batch):
    """
    straight head의 x(pred)와 bezier head의 x(pred)를 anchor row 단위로 맞추는 loss.

    outputs:
      'straight': { 'x': (B,L,R), ... }
      'bezier'  : { 'ctrl_points': (B,L,4,2), ... }
    batch:
      'gt_anchor': { 'mask': (B,L,R), ... }
    """
    device = outputs['bezier']['ctrl_points'].device

    ctrl = outputs['bezier']['ctrl_points'].detach()   # teacher, gradient 막음
    B, L, _, _ = ctrl.shape

    # straight branch output
    pred_x_st = outputs['straight']['x']               # (B,L,R)
    _, _, R = pred_x_st.shape

    # bezier를 R개 샘플링해서 straight의 row index와 align
    bezier_pts = sample_bezier(ctrl, num_samples=R)    # (B,L,R,2)
    pred_x_bz = bezier_pts[..., 0]                     # (B,L,R), x만

    mask = batch['gt_anchor']['mask'].to(device)       # (B,L,R), lane+row 유효 위치

    diff = torch.abs(pred_x_st - pred_x_bz) * mask     # (B,L,R)
    loss = diff.sum() / (mask.sum() + 1e-6)

    return loss

def bezier_x_on_rows(ctrl, row_ys, lane_mask=None, num_samples=100):
    """
    ctrl: (B,L,4,2) - bezier control points
    row_ys: (B,R)   - anchor row y (픽셀 좌표)
    lane_mask: (B,L) or None
    반환: x_bz (B,L,R) - 각 row_y에서의 bezier x
    """
    device = ctrl.device
    B, L, _, _ = ctrl.shape
    R = row_ys.shape[1]

    # bezier polyline 샘플링: (B,L,T,2)
    pts = sample_bezier(ctrl, num_samples=num_samples)      # (B,L,T,2)
    xs = pts[..., 0]                                        # (B,L,T)
    ys = pts[..., 1]                                        # (B,L,T)

    # 각 row_y에 대해 가장 가까운 y 인덱스 찾기 (vectorized)
    # ys_exp: (B,L,1,T)
    ys_exp = ys.unsqueeze(2)
    # row_ys_exp: (B,1,R,1)
    row_ys_exp = row_ys.view(B, 1, R, 1)
    # diff: (B,L,R,T)
    diff = torch.abs(ys_exp - row_ys_exp)
    idx = diff.argmin(dim=-1)                               # (B,L,R)

    # xs를 (B,L,R,T)로 확장해서 같은 위치에서 gather
    xs_exp = xs.unsqueeze(2).expand(-1, -1, R, -1)          # (B,L,R,T)
    idx_exp = idx.unsqueeze(-1)                             # (B,L,R,1)

    # T 축(dim=3)에서 gather
    x_bz = torch.gather(xs_exp, dim=3, index=idx_exp).squeeze(-1)  # (B,L,R)

    # lane_mask 있으면 invalid lane은 0으로
    if lane_mask is not None:
        lane_mask = lane_mask.to(device).view(B, L, 1)      # (B,L,1)
        x_bz = x_bz * lane_mask

    return x_bz

def routing_loss(outputs, batch, alpha_gate=0.1, tau=10.0):
    """
    outputs:
      'straight': { 'x': (B,L,R), 'exist_logit': ... }
      'bezier'  : { 'ctrl_points': (B,L,4,2), ... }
      'gate'    : (B,L,R) in [0,1]
    batch:
      'gt_anchor': { 'x','mask',... }
      'row_anchor_ys': (B,R)
      'lane_mask': (B,L)
    """
    device = outputs['bezier']['ctrl_points'].device

    x_s = outputs['straight']['x']                          # (B,L,R)
    ctrl = outputs['bezier']['ctrl_points']                 # (B,L,4,2)
    gate = outputs['gate']                                  # (B,L,R)

    gt_x = batch['gt_anchor']['x'].to(device)               # (B,L,R)
    mask = batch['gt_anchor']['mask'].to(device)            # (B,L,R)
    row_ys = batch['row_anchor_ys'].to(device).float()      # (B,R)
    lane_mask = batch['lane_mask'].to(device)               # (B,L)

    T = batch['gt_polyline'].shape[2]                       # GT polyline 길이 재사용

    # Bezier를 row_ys 위치로 보간해서 x_b 만들기
    x_b = bezier_x_on_rows(ctrl, row_ys, lane_mask=lane_mask, num_samples=T)  # (B,L,R)

    # 1) 최종 믹스 출력 x_mix = (1-g)*x_s + g*x_b
    x_mix = (1.0 - gate) * x_s + gate * x_b

    diff_mix = torch.abs(x_mix - gt_x) * mask
    L_mix = diff_mix.sum() / (mask.sum() + 1e-6)

    # 2) gate 힌트 loss (선택): e_s vs e_b 비교
    e_s = torch.abs(x_s - gt_x)
    e_b = torch.abs(x_b - gt_x)

    # e_b < e_s 이면 g_target > 0.5
    g_target = torch.sigmoid((e_s - e_b) / tau).detach()    # stop-grad

    bce = F.binary_cross_entropy(gate, g_target, reduction='none')
    L_gate = (bce * mask).sum() / (mask.sum() + 1e-6)

    L_total = L_mix + alpha_gate * L_gate
    return L_total, L_mix.detach(), L_gate.detach()

def compute_loss(outputs, batch, lambda_curve=1.0, lambda_cons=0.1, phase='curve_only'):
    device = outputs['bezier']['ctrl_points'].device

    loss_straight = torch.tensor(0.0, device=device)
    loss_curve = torch.tensor(0.0, device=device)
    loss_cons = torch.tensor(0.0, device=device)

    # curve loss
    if phase in ['curve_only', 'joint']:
        loss_curve = curve_loss(
            outputs['bezier']['ctrl_points'],
            batch['gt_polyline'].to(device),
            lane_mask=batch['lane_mask'].to(device),
            num_samples=batch['gt_polyline'].shape[2],
        )

    # straight loss
    if phase in ['straight_only', 'joint']:
        gt_anchor = {
            'x': batch['gt_anchor']['x'].to(device),
            'mask': batch['gt_anchor']['mask'].to(device),
            'exist': batch['gt_anchor']['exist'].to(device),
        }
        loss_straight = straight_loss(
            outputs['straight'],
            gt_anchor,
            lane_mask=batch['lane_mask'].to(device),
        )

    # consistency loss (Phase3)
    if phase == 'joint':
        loss_cons = consistency_loss(outputs, batch)

    # routing loss (Phase4)
    if phase == 'route':
        total_loss, L_mix, L_gate = routing_loss(outputs, batch)
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_straight': L_mix.item(),   # 로그에 mix를 straight slot에 찍자
            'loss_curve': 0.0,
            'loss_cons': L_gate.item(),      # gate loss를 cons slot에 찍자
        }
        return total_loss, loss_dict

    # 기존 phase들
    if phase == 'curve_only':
        total_loss = lambda_curve * loss_curve
    elif phase == 'straight_only':
        total_loss = loss_straight
    else:  # 'joint'
        total_loss = loss_straight + lambda_curve * loss_curve + lambda_cons * loss_cons

    loss_dict = {
        'loss_total': total_loss.item(),
        'loss_straight': loss_straight.item(),
        'loss_curve': loss_curve.item(),
        'loss_cons': loss_cons.item(),
    }
    return total_loss, loss_dict
