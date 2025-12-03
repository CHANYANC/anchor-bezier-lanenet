import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_dual import DualLaneDataset
from model_dual_head import DualHeadLaneNet
from bezier_utils import sample_bezier
from losses_dual import bezier_x_on_rows


def parse_args():
    """
    평가 스크립트용 인자 파서.
    - 4단계(Bezier / Anchor / Joint / Mix)의 체크포인트를 입력 받아
      동일한 방식으로 polyline L1 및 곡률 기반 smoothness를 계산한다.
    """
    p = argparse.ArgumentParser(description="Evaluate Bezier / Anchor / Mixed lane heads")

    p.add_argument('--data_root', type=str, required=True,
                   help='데이터셋 루트 디렉토리')
    p.add_argument('--list_val', type=str, required=True,
                   help='평가 이미지 리스트 파일')

    # 학습 단계별 체크포인트
    p.add_argument('--ckpt_phase1', type=str, required=True,
                   help='Phase1 (curve-only / Bezier head) 체크포인트')
    p.add_argument('--ckpt_phase2', type=str, required=True,
                   help='Phase2 (straight-only / anchor head) 체크포인트')
    p.add_argument('--ckpt_phase3', type=str, required=True,
                   help='Phase3 (joint anchor) 체크포인트')
    p.add_argument('--ckpt_phase4', type=str, required=True,
                   help='Phase4 (router / mixed head) 체크포인트')

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)

    return p.parse_args()


def build_model_phase1(ckpt_path, device):
    """
    Phase1 전용 로더.
    - 예전 체크포인트는 neck_bezier.*, head_bezier.* 이름을 사용하기 때문에
      현재 neck_curve.*, head_curve.* 로 맵핑해준다.
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

        # 이름도 맞고 shape도 맞을 때만 로드
        if new_k in model_state and model_state[new_k].shape == v.shape:
            mapped[new_k] = v

    model_state.update(mapped)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def build_model_general(ckpt_path, device):
    """
    Phase2 / Phase3 / Phase4 공통 로더.
    - 체크포인트 구성과 모델 구조가 일치한다고 가정한다.
    """
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
    2차 미분 기반 lane smoothness 계산.
    - |x[t+1] - 2 * x[t] + x[t-1]| 의 평균값을 반환.
    - 값이 작을수록 더 부드러운 곡선.
    """
    if x_seq.numel() < 3:
        return 0.0
    d2 = x_seq[2:] - 2 * x_seq[1:-1] + x_seq[:-2]
    return d2.abs().mean().item()


@torch.no_grad()
def eval_polyline_bezier(model, loader, device):
    """
    Bezier head 전용 polyline 평가:
    - GT polyline과 Bezier sample polyline 간 L1
    - 예측/GT 곡선의 smoothness 비교
    """
    lane_l1 = []
    lane_smooth_pred = []
    lane_smooth_gt = []

    for batch in loader:
        images = batch['image'].to(device)
        gt_poly = batch['gt_polyline'].to(device)     # (B,L,T,2)
        lane_mask = batch['lane_mask'].to(device)     # (B,L)

        outputs = model(images)
        ctrl = outputs['bezier']['ctrl_points']       # (B,L,4,2)

        B, L, T, _ = gt_poly.shape
        pred_poly = sample_bezier(ctrl, num_samples=T)

        diff = (pred_poly - gt_poly).abs()
        diff_sum = diff[..., 0].sum(dim=2) + diff[..., 1].sum(dim=2)
        lane_l1_batch = diff_sum / (T + 1e-6)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue

                lane_l1.append(lane_l1_batch[b, l].item())

                x_pred = pred_poly[b, l, :, 0]
                x_gt = gt_poly[b, l, :, 0]
                lane_smooth_pred.append(second_diff_smoothness(x_pred))
                lane_smooth_gt.append(second_diff_smoothness(x_gt))

    return np.array(lane_l1), np.array(lane_smooth_pred), np.array(lane_smooth_gt)


@torch.no_grad()
def eval_polyline_anchor_or_mix(model, loader, device, mode: str):
    """
    Anchor 기반 또는 Router(Mix) 기반 polyline 평가.
    mode='anchor' → anchor head 값 사용
    mode='mix'    → (1-g)*anchor + g*bezier_row

    - row anchor y 위치에 따라 x를 재구성하여 polyline 형태로 비교한다.
    """
    assert mode in ['anchor', 'mix']

    lane_l1 = []
    lane_smooth_pred = []
    lane_smooth_gt = []

    for batch in loader:
        images = batch['image'].to(device)
        gt_poly = batch['gt_polyline'].to(device)
        lane_mask = batch['lane_mask'].to(device)
        row_ys = batch['row_anchor_ys'].to(device).float()  # (B,R)

        outputs = model(images)
        x_anchor = outputs['straight']['x']  # (B,L,R)

        if mode == 'anchor':
            x_used = x_anchor

        else:
            # gate + Bezier row projection
            ctrl = outputs['bezier']['ctrl_points']
            gate = outputs['gate']

            x_bezier_rows = bezier_x_on_rows(
                ctrl, row_ys, lane_mask=lane_mask, num_samples=gt_poly.shape[2]
            )

            x_used = (1.0 - gate) * x_anchor + gate * x_bezier_rows

        # polyline 재구성
        B, L, T, _ = gt_poly.shape
        pred_poly = torch.zeros_like(gt_poly)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue

                for t in range(T):
                    y_gt = gt_poly[b, l, t, 1]
                    idx = torch.argmin(torch.abs(row_ys[b] - y_gt))
                    pred_poly[b, l, t, 0] = x_used[b, l, idx]
                    pred_poly[b, l, t, 1] = y_gt

        diff = (pred_poly - gt_poly).abs()
        diff_sum = diff[..., 0].sum(dim=2) + diff[..., 1].sum(dim=2)
        lane_l1_batch = diff_sum / (T + 1e-6)

        for b in range(B):
            for l in range(L):
                if lane_mask[b, l] < 0.5:
                    continue

                lane_l1.append(lane_l1_batch[b, l].item())

                x_pred = pred_poly[b, l, :, 0]
                x_gt = gt_poly[b, l, :, 0]
                lane_smooth_pred.append(second_diff_smoothness(x_pred))
                lane_smooth_gt.append(second_diff_smoothness(x_gt))

    return np.array(lane_l1), np.array(lane_smooth_pred), np.array(lane_smooth_gt)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 공통 validation dataloader
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
    model_p1 = build_model_phase1(args.ckpt_phase1, device)
    model_p2 = build_model_general(args.ckpt_phase2, device)
    model_p3 = build_model_general(args.ckpt_phase3, device)
    model_p4 = build_model_general(args.ckpt_phase4, device)

    # 평가 수행
    l1_b, s_b_pred, s_b_gt = eval_polyline_bezier(model_p1, loader, device)
    l1_a2, s_a2_pred, s_a2_gt = eval_polyline_anchor_or_mix(model_p2, loader, device, mode="anchor")
    l1_a3, s_a3_pred, s_a3_gt = eval_polyline_anchor_or_mix(model_p3, loader, device, mode="anchor")
    l1_m4, s_m4_pred, s_m4_gt = eval_polyline_anchor_or_mix(model_p4, loader, device, mode="mix")

    smooth_gt_mean = np.mean(s_b_gt) if s_b_gt.size > 0 else 0.0

    print("=====================================")
    print("[Polyline L1 vs GT] (per-lane)")
    print(f"P1 Bezier : mean={l1_b.mean():.3f}, std={l1_b.std():.3f}, num_lanes={l1_b.shape[0]}")
    print(f"P2 Anchor : mean={l1_a2.mean():.3f}, std={l1_a2.std():.3f}, num_lanes={l1_a2.shape[0]}")
    print(f"P3 Anchor : mean={l1_a3.mean():.3f}, std={l1_a3.std():.3f}, num_lanes={l1_a3.shape[0]}")
    print(f"P4 Mix    : mean={l1_m4.mean():.3f}, std={l1_m4.std():.3f}, num_lanes={l1_m4.shape[0]}")
    print("=====================================")
    print("[Smoothness (Δ² x, per-lane)]")
    print(f"GT        : mean={smooth_gt_mean:.6f}")
    print(f"P1 Bezier : mean={s_b_pred.mean():.6f}")
    print(f"P2 Anchor : mean={s_a2_pred.mean():.6f}")
    print(f"P3 Anchor : mean={s_a3_pred.mean():.6f}")
    print(f"P4 Mix    : mean={s_m4_pred.mean():.6f}")
    print("=====================================")


if __name__ == '__main__':
    main()
