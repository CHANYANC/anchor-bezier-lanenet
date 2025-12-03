import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DualLaneDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        list_path: str,
        num_samples: int = 40,
        num_lanes: int = 4,
    ):
        self.data_root = data_root
        self.num_samples = num_samples
        self.num_lanes = num_lanes

        # CULane list.txt 로부터 이미지 상대 경로 목록 읽기
        with open(list_path, 'r') as f:
            self.lines = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.lines)

    def _load_image(self, img_path: str) -> torch.Tensor:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"[DualLaneDataset] 이미지 로드 실패: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)
        return img

    def _resample_lane(self, pts: np.ndarray) -> np.ndarray:
        T = self.num_samples
        N = pts.shape[0]

        if N == 0:
            return np.zeros((T, 2), dtype=np.float32)

        if N == T:
            return pts.astype(np.float32)

        if N > T:
            # N개 중에서 T개를 균일 간격으로 샘플링
            idx = np.linspace(0, N - 1, T)
            idx = np.round(idx).astype(int)
            pts_res = pts[idx]
        else:
            # 부족하면 마지막 점을 반복해서 pad
            pad = T - N
            last = pts[-1:, :]  # (1,2)
            pad_pts = np.repeat(last, pad, axis=0)  # (pad,2)
            pts_res = np.concatenate([pts, pad_pts], axis=0)

        return pts_res.astype(np.float32)

    def _load_polyline_from_lines(
        self,
        img_path: str,
        H: int,
        W: int,
    ):

        # 기본 가정: label 파일 경로 = 이미지 경로에서 확장자만 .lines.txt로 변경
        # 예) .../00000.jpg -> .../00000.lines.txt
        base, ext = os.path.splitext(img_path)
        lines_path = base + '.lines.txt'

        if not os.path.isfile(lines_path):
            # 라벨 없으면 전부 0으로
            L = self.num_lanes
            T = self.num_samples
            gt_polyline = torch.zeros(L, T, 2, dtype=torch.float32)
            lane_mask = torch.zeros(L, dtype=torch.float32)
            return gt_polyline, lane_mask

        lane_list = []
        masks = []

        with open(lines_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                vals = line.split()
                if len(vals) < 4:
                    continue

                coords = np.array([float(v) for v in vals], dtype=np.float32)
                pts = coords.reshape(-1, 2)  # (N,2) = (x,y)

                # x<0, y<0 같은 비유효 포인트 제거 (CULane 관례)
                valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
                pts = pts[valid]

                if pts.shape[0] == 0:
                    continue

                # (T,2)로 resample
                pts_res = self._resample_lane(pts)  # (T,2)
                lane_list.append(torch.from_numpy(pts_res).unsqueeze(0))  # (1,T,2)
                masks.append(1.0)

        # lane 수를 num_lanes로 맞추기
        while len(lane_list) < self.num_lanes:
            lane_list.append(torch.zeros(1, self.num_samples, 2, dtype=torch.float32))
            masks.append(0.0)

        lane_list = lane_list[:self.num_lanes]
        masks = masks[:self.num_lanes]

        gt_polyline = torch.cat(lane_list, dim=0)  # (L,T,2)
        lane_mask = torch.tensor(masks, dtype=torch.float32)  # (L,)

        return gt_polyline, lane_mask


    def _build_anchor_gt(
        self,
        gt_polyline: torch.Tensor,   # (L,T,2) 픽셀
        lane_mask: torch.Tensor,     # (L,)
        row_anchor_ys: torch.Tensor, # (R,)
        img_w: int,
    ):
        L, T, _ = gt_polyline.shape
        R = row_anchor_ys.shape[0]

        gt_x = torch.zeros(L, R, dtype=torch.float32)
        gt_mask = torch.zeros(L, R, dtype=torch.float32)

        poly = gt_polyline.numpy()               # (L,T,2)
        lane_mask_np = lane_mask.numpy()         # (L,)
        row_ys_np = row_anchor_ys.numpy().astype(float)  # (R,)

        for l in range(L):
            if lane_mask_np[l] < 0.5:
                continue

            pts = poly[l]       # (T,2)
            xs = pts[:, 0]
            ys = pts[:, 1]

            # y 기준 정렬
            order = np.argsort(ys)
            xs = xs[order]
            ys = ys[order]

            for r in range(R):
                y0 = row_ys_np[r]
                found = False

                # y0가 [y1,y2] 구간에 들어가는 segment 찾기
                for i in range(len(ys) - 1):
                    y1, y2 = ys[i], ys[i + 1]
                    x1, x2 = xs[i], xs[i + 1]

                    if (y1 - y0) * (y2 - y0) <= 0 and (y1 != y2):
                        t = (y0 - y1) / (y2 - y1)
                        x_interp = x1 + t * (x2 - x1)
                        gt_x[l, r] = float(x_interp)
                        gt_mask[l, r] = 1.0
                        found = True
                        break

                if not found:
                    gt_mask[l, r] = 0.0

        gt_x = torch.clamp(gt_x, 0.0, float(img_w - 1))
        return gt_x, gt_mask

    def __getitem__(self, idx: int):
        rel_path = self.lines[idx]
        img_path = os.path.join(self.data_root, rel_path.lstrip('/'))

        image = self._load_image(img_path)  # (3,H,W)
        H, W = image.shape[1], image.shape[2]

        gt_polyline, lane_mask = self._load_polyline_from_lines(img_path, H, W)

        # row_anchor_ys: H 기준 uniform
        R = 40
        row_anchor_ys = torch.linspace(0, H - 1, R).long()

        # straight head용 anchor GT
        gt_x, gt_mask = self._build_anchor_gt(gt_polyline, lane_mask, row_anchor_ys, W)
        gt_anchor = {
            'x': gt_x,            # (L,R)
            'mask': gt_mask,      # (L,R)
            'exist': lane_mask.clone(),  # (L,)
        }

        sample = {
            'image': image,
            'gt_anchor': gt_anchor,
            'gt_polyline': gt_polyline,
            'lane_mask': lane_mask,
            'row_anchor_ys': row_anchor_ys,
            'meta': {
                'img_h': H,
                'img_w': W,
                'path': img_path,
            },
        }
        return sample
