import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_dual import DualLaneDataset
from model_dual_head import DualHeadLaneNet
from bezier_utils import sample_bezier
from losses_dual import bezier_x_on_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--list_val', type=str, required=True)

    # P1: Bezier (Phase1 curve_only)
    p.add_argument('--ckpt_phase1', type=str, required=True)
    # P2: Anchor (Phase2 straight_only)
    p.add_argument('--ckpt_phase2', type=str, required=True)
    # P3: Anchor (Phase3 joint)
    p.add_argument('--ckpt_phase3', type=str, required=True)
    # P4: Mix (Phase4 route)
    p.add_argument('--ckpt_phase4', type=str, required=True)

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def build_model_phase1(ckpt_path, device):
    """
    Phase1(curve_only) ckpt 로딩:
      - 옛날 neck_bezier/head_bezier 이름을 neck_curve/head_curve로 매핑.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_state = ckpt['model']

    model = DualHeadLaneNet(
        num_lanes=4,
        num_rows=40,
        num_cols=72,
        num_cell_row=100,
        num_cell_col=100,
    )
    model_state = model.state_dict()

    mapped = {}
    for k, v in ckpt_state.items():
        new_k = k
        if k.startswith('neck_bezier.'):
            new_k = 'neck_curve.' + k[len('neck_bezier.'):]
        elif k.startswith('head_bezier.'):
            new_k = 'head_curve.' + k[len('head_bezier.'):]

        if new_k in model_state and model_state[new_k].shape == v.shape:
            mapped[new_k] = v

    model_state.update(mapped)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def build_model_general(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = DualHeadLaneNet(
        num_lanes=4,
        num_rows=40,
        num_cols=72,
        num_cell_row=100,
        num_cell_col=100,
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    return model


def second_diff_smoothness(x_seq: torch.Tensor) -> float:
    """
    x_seq: (T,) 1D 텐서
    반환: mean(|Δ² x|) 스칼라 (float)
    """
    if x_seq.numel() < 3:
        return 0.0
    d2 = x_seq[2:] - 2 * x_seq[1:-1] + x_seq[:-2]
    return d2.abs().mean().item()


@torch.no_grad()
def eval_polyline_bezier(model, loader, device):
    """
    P1: Bezier head
      - polyline L1: sample_bezier(ctrl, T) vs gt_polyline
      - smoothness: pred polyline / gt polyline Δ²x
    """
    lane_l1 = []
    lane_smooth_pred = []
    lane_smooth_gt = []

    for batch in loader:
        images = batch['image'].to(device)
        gt_poly = batch['gt_polyline'].to(device)      # (B,L,T,2)
        lane_mask = batch['lane_mask'].to(device)      # (B,L)
        B, L, T, _ = gt_poly.shape

        outputs = model(images)
        ctrl = outputs['bezier']['ctrl_points']        # (B,L,4,2)
        pred_poly = sample_bezier(ctrl, num_samples=T) # (B,L,T,2)

        diff = (pred_poly - gt_poly).abs()             # (B,L,T,2)
        # x,y 둘 다 합친 L1
        diff_sum = diff[..., 0].sum(dim=2) + diff[..., 1].sum(dim=2)  # (B,L)
        lane_len = torch.ones_like(diff_sum) * T

        lane_l1_batch = diff_sum / (lane_len + 1e-6)   # (B,L)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue
                lane_l1.append(lane_l1_batch[b, l].item())

                x_pred = pred_poly[b, l, :, 0]
                x_gt = gt_poly[b, l, :, 0]
                lane_smooth_pred.append(second_diff_smoothness(x_pred))
                lane_smooth_gt.append(second_diff_smoothness(x_gt))

    lane_l1 = np.array(lane_l1) if lane_l1 else np.array([])
    lane_smooth_pred = np.array(lane_smooth_pred) if lane_smooth_pred else np.array([])
    lane_smooth_gt = np.array(lane_smooth_gt) if lane_smooth_gt else np.array([])

    return lane_l1, lane_smooth_pred, lane_smooth_gt


@torch.no_grad()
def eval_polyline_anchor_or_mix(model, loader, device, mode: str):
    """
    P2/P3/P4: Anchor or Mix
      - mode='anchor': anchor x만 사용 (straight head)
      - mode='mix'   : mix = (1-g)*anchor + g*Bezier_row
    polyline 구성:
      - GT polyline의 각 y 위치에서
      - 가장 가까운 row anchor index 찾아서
      - 그 row에서의 x_used 값을 가져와 polyline 생성.
    """
    assert mode in ['anchor', 'mix']
    lane_l1 = []
    lane_smooth_pred = []
    lane_smooth_gt = []

    for batch in loader:
        images = batch['image'].to(device)
        gt_poly = batch['gt_polyline'].to(device)         # (B,L,T,2)
        lane_mask = batch['lane_mask'].to(device)         # (B,L)
        row_ys = batch['row_anchor_ys'].to(device).float()  # (B,R)

        B, L, T, _ = gt_poly.shape
        R = row_ys.shape[1]

        outputs = model(images)
        x_anchor = outputs['straight']['x']               # (B,L,R)

        if mode == 'anchor':
            x_used = x_anchor                             # (B,L,R)
        else:  # 'mix'
            ctrl = outputs['bezier']['ctrl_points']       # (B,L,4,2)
            gate = outputs['gate']                        # (B,L,R)
            # Bezier를 row anchor 위치로 투영
            T_row = gt_poly.shape[2]
            x_b = bezier_x_on_rows(
                ctrl,
                row_ys,
                lane_mask=lane_mask,
                num_samples=T_row
            )                                             # (B,L,R)
            x_used = (1.0 - gate) * x_anchor + gate * x_b # (B,L,R)

        # pred polyline: (B,L,T,2)
        pred_poly = torch.zeros_like(gt_poly)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue

                for t in range(T):
                    y_gt = gt_poly[b, l, t, 1]
                    # 가장 가까운 row anchor index
                    diff_y = torch.abs(row_ys[b] - y_gt)  # (R,)
                    idx = torch.argmin(diff_y)
                    x_val = x_used[b, l, idx]

                    pred_poly[b, l, t, 0] = x_val
                    pred_poly[b, l, t, 1] = y_gt

        diff = (pred_poly - gt_poly).abs()                # (B,L,T,2)
        diff_sum = diff[..., 0].sum(dim=2) + diff[..., 1].sum(dim=2)  # (B,L)
        lane_len = torch.ones_like(diff_sum) * T

        lane_l1_batch = diff_sum / (lane_len + 1e-6)      # (B,L)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue
                lane_l1.append(lane_l1_batch[b, l].item())

                x_pred = pred_poly[b, l, :, 0]
                x_gt = gt_poly[b, l, :, 0]
                lane_smooth_pred.append(second_diff_smoothness(x_pred))
                lane_smooth_gt.append(second_diff_smoothness(x_gt))

    lane_l1 = np.array(lane_l1) if lane_l1 else np.array([])
    lane_smooth_pred = np.array(lane_smooth_pred) if lane_smooth_pred else np.array([])
    lane_smooth_gt = np.array(lane_smooth_gt) if lane_smooth_gt else np.array([])

    return lane_l1, lane_smooth_pred, lane_smooth_gt


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 공통 val loader
    dataset = DualLaneDataset(
        data_root=args.data_root,
        list_path=args.list_val,
        num_samples=40,
        num_lanes=4,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 모델 로드
    model_p1 = build_model_phase1(args.ckpt_phase1, device)   # Bezier
    model_p2 = build_model_general(args.ckpt_phase2, device)  # Anchor (Phase2)
    model_p3 = build_model_general(args.ckpt_phase3, device)  # Anchor (Phase3 joint)
    model_p4 = build_model_general(args.ckpt_phase4, device)  # Mix   (Phase4 route)

    # 1) P1: Bezier polyline metrics
    l1_b, smooth_b_pred, smooth_gt_b = eval_polyline_bezier(model_p1, loader, device)

    # 2) P2: Anchor polyline metrics
    l1_a2, smooth_a2_pred, smooth_gt_a2 = eval_polyline_anchor_or_mix(
        model_p2, loader, device, mode='anchor'
    )

    # 3) P3: Anchor polyline metrics (joint)
    l1_a3, smooth_a3_pred, smooth_gt_a3 = eval_polyline_anchor_or_mix(
        model_p3, loader, device, mode='anchor'
    )

    # 4) P4: Mix polyline metrics (route)
    l1_m4, smooth_m4_pred, smooth_gt_m4 = eval_polyline_anchor_or_mix(
        model_p4, loader, device, mode='mix'
    )

    # GT smoothness는 모두 동일해야 하니, 하나 골라서 사용
    smooth_gt_mean = np.mean(smooth_gt_b) if smooth_gt_b.size > 0 else 0.0

    print("=====================================")
    print("[Polyline L1 vs GT] (per-lane)")
    print(f"P1 Bezier : mean={l1_b.mean():.3f}, std={l1_b.std():.3f}, num_lanes={l1_b.shape[0]}")
    print(f"P2 Anchor : mean={l1_a2.mean():.3f}, std={l1_a2.std():.3f}, num_lanes={l1_a2.shape[0]}")
    print(f"P3 Anchor : mean={l1_a3.mean():.3f}, std={l1_a3.std():.3f}, num_lanes={l1_a3.shape[0]}")
    print(f"P4 Mix    : mean={l1_m4.mean():.3f}, std={l1_m4.std():.3f}, num_lanes={l1_m4.shape[0]}")
    print("=====================================")
    print("[Smoothness (Δ² x, per-lane)]")
    print(f"GT        : mean={smooth_gt_mean:.6f}")
    print(f"P1 Bezier : mean={smooth_b_pred.mean():.6f}")
    print(f"P2 Anchor : mean={smooth_a2_pred.mean():.6f}")
    print(f"P3 Anchor : mean={smooth_a3_pred.mean():.6f}")
    print(f"P4 Mix    : mean={smooth_m4_pred.mean():.6f}")
    print("=====================================")


if __name__ == '__main__':
    main()
