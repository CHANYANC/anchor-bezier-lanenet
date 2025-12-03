# train_dual.py
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from model_dual_head import DualHeadLaneNet
from losses_dual import compute_loss
from dataset_dual import DualLaneDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--list_train', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_curve', type=float, default=1.0)
    parser.add_argument('--lambda_cons', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./runs_dual')

    parser.add_argument('--phase', type=str, default='curve_only',
                        choices=['curve_only', 'straight_only', 'joint', 'route'])


    return parser.parse_args()


def build_dataloader(args):
    # .lines.txt 기반 DualLaneDataset: bezier_json 안 씀
    dataset = DualLaneDataset(
        data_root=args.data_root,
        list_path=args.list_train,
        num_samples=40,
        num_lanes=4,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader

def freeze_for_phase(model, phase):
    # 기본: 전부 train 가능
    for p in model.parameters():
        p.requires_grad = True

    if phase == 'curve_only':
        # straight branch freeze
        for p in model.neck_straight.parameters():
            p.requires_grad = False
        for p in model.head_straight.parameters():
            p.requires_grad = False

    elif phase == 'straight_only':
        # bezier branch freeze
        for p in model.neck_curve.parameters():
            p.requires_grad = False
        for p in model.head_curve.parameters():
            p.requires_grad = False

    elif phase == 'joint':
        # 현재처럼 두 expert 다 학습 (또는 필요하면 bezier를 teacher로 얼려도 되고)
        pass

    elif phase == 'route':
        # backbone + 두 expert + curve/straight neck/heads 전부 freeze
        for p in model.backbone.parameters():
            p.requires_grad = False

        for p in model.neck_straight.parameters():
            p.requires_grad = False
        for p in model.head_straight.parameters():
            p.requires_grad = False

        for p in model.neck_curve.parameters():
            p.requires_grad = False
        for p in model.head_curve.parameters():
            p.requires_grad = False

        # gate 쪽(head_gate / neck_gate 등)만 trainable 상태로 남김
        # (RoutingHead 안의 파라미터들은 그대로 requires_grad=True라서 학습됨)

    return model


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DualHeadLaneNet(
        num_lanes=4,
        num_rows=40,
        num_cols=72,
        num_cell_row=100,
        num_cell_col=100,
    )
    model.to(device)

    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resume from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        ckpt_state = ckpt['model']
        model_state = model.state_dict()

        mapped = {}
        for k, v in ckpt_state.items():
            new_k = k

            # Phase1 ckpt의 bezier 브랜치 이름을 새 구조로 매핑
            if k.startswith('neck_bezier.'):
                new_k = 'neck_curve.' + k[len('neck_bezier.'):]
            elif k.startswith('head_bezier.'):
                new_k = 'head_curve.' + k[len('head_bezier.'):]

            # 나머지(backbone 등)는 이름 그대로 두고 shape 맞는 것만 사용
            if new_k in model_state and model_state[new_k].shape == v.shape:
                mapped[new_k] = v

        print(f"[INFO] Load {len(mapped)} tensors from ckpt into current model")
        model_state.update(mapped)
        model.load_state_dict(model_state, strict=False)

    model = freeze_for_phase(model, args.phase)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    loader = build_dataloader(args)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            images = batch['image'].to(device)

            batch['gt_polyline'] = batch['gt_polyline'].to(device)
            if 'lane_mask' in batch and batch['lane_mask'] is not None:
                batch['lane_mask'] = batch['lane_mask'].to(device)
            batch['row_anchor_ys'] = batch['row_anchor_ys'].to(device)

            outputs = model(images)

            loss, loss_dict = compute_loss(
                outputs,
                batch,
                lambda_curve=args.lambda_curve,
                lambda_cons=args.lambda_cons,
                phase=args.phase,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(f"[Iter {global_step}] "
                      f"loss={loss.item():.6f}, "
                      f"straight={loss_dict['loss_straight']:.6f}, "
                      f"curve={loss_dict['loss_curve']:.6f}, "
                      f"cons={loss_dict['loss_cons']:.6f}")

        epoch_loss /= len(loader)
        dt = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"phase={args.phase}, "
              f"loss={epoch_loss:.6f}, time={dt:.1f}s")

        save_path = os.path.join(
            args.save_dir,
            f"dual_{args.phase}_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'args': vars(args),
            },
            save_path
        )
        print(f"[SAVE] {save_path}")


if __name__ == '__main__':
    main()
