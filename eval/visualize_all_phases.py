import os
import argparse

import numpy as np
import cv2
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

    p.add_argument('--ckpt_phase1', type=str, required=True)  # curve_only (Bezier)
    p.add_argument('--ckpt_phase2', type=str, required=True)  # straight_only (anchor)
    p.add_argument('--ckpt_phase3', type=str, required=True)  # joint (anchor)
    p.add_argument('--ckpt_phase4', type=str, required=True)  # route (mix)

    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_vis', type=int, default=20)         # 몇 장만 뽑을지
    p.add_argument('--max_err_for_cmap', type=float, default=60.0)  # P4 에러 그라데이션 상한(px)
    return p.parse_args()


def build_model_phase1(ckpt_path, device):
    """
    Phase1(curve_only) ckpt:
      - neck_bezier/head_bezier -> neck_curve/head_curve 매핑해서 로딩.
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


def tensor_to_image(img_tensor):
    """
    img_tensor: (3,H,W), 0~1 가정
    -> BGR uint8 이미지
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # H,W,C
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def clamp_xy(x, y, W, H):
    x = int(round(float(x)))
    y = int(round(float(y)))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y


def draw_filled_circle(img, x, y, color, radius=3):
    H, W = img.shape[:2]
    x, y = clamp_xy(x, y, W, H)
    cv2.circle(img, (x, y), radius, color, thickness=-1)
    return img


def draw_filled_triangle(img, x, y, color, size=4):
    """
    채워진 삼각형 (P2)
    """
    H, W = img.shape[:2]
    x, y = clamp_xy(x, y, W, H)
    pts = np.array([
        [x, y - size],
        [x - size, y + size],
        [x + size, y + size]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    return img


def draw_filled_square(img, x, y, color, size=4):
    """
    채워진 네모 (P3)
    """
    H, W = img.shape[:2]
    x, y = clamp_xy(x, y, W, H)
    pts = np.array([
        [x - size, y - size],
        [x + size, y - size],
        [x + size, y + size],
        [x - size, y + size]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    return img


def draw_filled_star(img, x, y, color, outer_radius=6, inner_radius=3):
    """
    5각 별 (채워진) – P4 시각화용
    """
    H, W = img.shape[:2]
    x, y = clamp_xy(x, y, W, H)

    pts = []
    for i in range(10):
        angle = i * (np.pi / 5.0) - np.pi / 2.0  # -90도 시작
        r = outer_radius if i % 2 == 0 else inner_radius
        xi = x + int(r * np.cos(angle))
        yi = y + int(r * np.sin(angle))
        pts.append([xi, yi])

    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    return img


def draw_polyline(img, xs, ys, color, thickness=2):
    H, W = img.shape[:2]
    pts = []
    for x, y in zip(xs, ys):
        x, y = clamp_xy(x, y, W, H)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    return img


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    # dataset / loader
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

    # models
    model_p1 = build_model_phase1(args.ckpt_phase1, device)
    model_p2 = build_model_general(args.ckpt_phase2, device)
    model_p3 = build_model_general(args.ckpt_phase3, device)
    model_p4 = build_model_general(args.ckpt_phase4, device)

    cnt = 0
    for batch in loader:
        if cnt >= args.num_vis:
            break

        images = batch['image'].to(device)  # (B,3,H,W)
        B, _, H, W = images.shape

        row_ys_all = batch['row_anchor_ys']           # (B,R)
        gt_anchor = batch['gt_anchor']               # dict of (B,L,R)
        lane_mask_all = batch['lane_mask']           # (B,L)
        meta = batch['meta']
        gt_poly = batch['gt_polyline']               # (B,L,T,2)
        T = gt_poly.shape[2]

        # forward
        out1 = model_p1(images)
        out2 = model_p2(images)
        out3 = model_p3(images)
        out4 = model_p4(images)

        for b in range(B):
            if cnt >= args.num_vis:
                break

            img_tensor = batch['image'][b]
            img_vis = tensor_to_image(img_tensor)

            row_ys = row_ys_all[b].to(device).float()      # (R,)
            lane_mask = lane_mask_all[b].to(device)        # (L,)

            x_gt = gt_anchor['x'][b].cpu().numpy()         # (L,R)
            mask_gt = gt_anchor['mask'][b].cpu().numpy()   # (L,R)
            L, R = x_gt.shape

            # ---- P1: Bezier polyline (샘플링해서 (x,y) 곡선 그리기) ----
            ctrl1 = out1['bezier']['ctrl_points'][b]       # (L,4,2)
            bezier_poly = sample_bezier(
                ctrl1.unsqueeze(0), num_samples=T
            )[0].cpu().numpy()  # (L,T,2)

            # ---- P2 / P3: straight x ----
            x_p2 = out2['straight']['x'][b].cpu().numpy()  # (L,R)
            x_p3 = out3['straight']['x'][b].cpu().numpy()  # (L,R)

            # ---- P4: mix = (1-g)*x_s + g*x_b ----
            x_s_p4 = out4['straight']['x'][b]              # (L,R)
            ctrl4 = out4['bezier']['ctrl_points'][b:b+1]   # (1,L,4,2)
            gate4 = out4['gate'][b]                        # (L,R)
            x_b_p4 = bezier_x_on_rows(
                ctrl4,
                row_ys.unsqueeze(0),
                lane_mask=lane_mask.unsqueeze(0),
                num_samples=T,
            )[0]                                           # (L,R)
            x_mix_p4 = (1.0 - gate4) * x_s_p4 + gate4 * x_b_p4
            x_mix_p4 = x_mix_p4.cpu().numpy()

            # clamp
            x_gt_cl = np.clip(x_gt, 0.0, float(W - 1))
            x_p2 = np.clip(x_p2, 0.0, float(W - 1))
            x_p3 = np.clip(x_p3, 0.0, float(W - 1))
            x_mix_p4 = np.clip(x_mix_p4, 0.0, float(W - 1))

            row_ys_np = row_ys.cpu().numpy().astype(float)
            row_ys_np = np.clip(row_ys_np, 0.0, float(H - 1))

            for l in range(L):
                if lane_mask[l] < 0.5:
                    continue

                valid = mask_gt[l] > 0.5
                idxs = np.where(valid)[0]
                if len(idxs) == 0:
                    continue

                ys_lane = row_ys_np[idxs]
                x_gt_lane = x_gt_cl[l, idxs]
                x_p2_lane = x_p2[l, idxs]
                x_p3_lane = x_p3[l, idxs]
                x_p4_lane = x_mix_p4[l, idxs]

                # y 기준 정렬
                order = np.argsort(ys_lane)
                ys_lane = ys_lane[order]
                x_gt_lane = x_gt_lane[order]
                x_p2_lane = x_p2_lane[order]
                x_p3_lane = x_p3_lane[order]
                x_p4_lane = x_p4_lane[order]

                # 1) GT: 빨간 점 (채워진 동그라미)
                for xg, yg in zip(x_gt_lane, ys_lane):
                    img_vis = draw_filled_circle(
                        img_vis, xg, yg,
                        color=(0, 0, 255),  # BGR: red
                        radius=3,
                    )

                # 2) P1: Bezier polyline (파란 계열 선)
                bezier_lane = bezier_poly[l]  # (T,2)
                xs_b = bezier_lane[:, 0]
                ys_b = bezier_lane[:, 1]
                img_vis = draw_polyline(
                    img_vis,
                    xs_b,
                    ys_b,
                    color=(255, 0, 0),  # BGR: blue-ish
                    thickness=2
                )

                # 3) P2: 채워진 삼각형 (Cyan)
                for xp2, yp2 in zip(x_p2_lane, ys_lane):
                    img_vis = draw_filled_triangle(
                        img_vis, xp2, yp2,
                        color=(255, 255, 0),  # cyan-ish
                        size=4,
                    )

                # 4) P3: 채워진 네모 (초록)
                for xp3, yp3 in zip(x_p3_lane, ys_lane):
                    img_vis = draw_filled_square(
                        img_vis, xp3, yp3,
                        color=(0, 255, 0),  # green
                        size=4,
                    )

                # 5) P4: 별 (노란색 계열 + 에러 그라데이션)
                err = np.abs(x_p4_lane - x_gt_lane)  # (N,)
                norm_err = np.clip(err / args.max_err_for_cmap, 0.0, 1.0)
                for xp4, yp4, ne in zip(x_p4_lane, ys_lane, norm_err):
                    # ne=0 (작은 에러) -> 밝은 노랑, ne=1 (큰 에러) -> 어두운 노랑/주황
                    alpha = 1.0 - ne  # 1: good, 0: bad
                    g = int(150 + 105 * alpha)
                    r = int(150 + 105 * alpha)
                    b = int(30 * alpha)
                    color = (b, g, r)   # BGR: yellow 계열
                    img_vis = draw_filled_star(
                        img_vis, xp4, yp4,
                        color=color,
                        outer_radius=6,
                        inner_radius=3,
                    )

            # 이미지 이름 결정 (meta['path'] 구조에 상관없이 안전하게)
            if isinstance(meta, dict) and 'path' in meta:
                path_info = meta['path']
                if isinstance(path_info, (list, tuple)):
                    if len(path_info) > b:
                        img_path = path_info[b]
                    elif len(path_info) > 0:
                        img_path = path_info[0]
                    else:
                        img_path = f"img_{cnt:05d}.png"
                else:
                    img_path = path_info
            else:
                img_path = f"img_{cnt:05d}.png"

            base_name = os.path.basename(str(img_path))
            name_wo_ext, _ = os.path.splitext(base_name)
            save_path = os.path.join(
                args.out_dir,
                f"{cnt:05d}_{name_wo_ext}_compare.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_vis)
            print(f"[SAVE] {save_path}")
            cnt += 1
            if cnt >= args.num_vis:
                break


if __name__ == '__main__':
    main()
