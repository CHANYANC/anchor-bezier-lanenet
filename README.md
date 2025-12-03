# Anchor와 Bezier 곡선을 결합한 Dual-Head구조의 차선 인식 모델

이 레포는 **같은 차선 정보를 두 가지 표현(Anchor / Bezier)으로 잡고, 상황에 따라 어느 쪽을 더 신뢰할지 라우팅하는 구조**를 구현한다.

- **Anchor head**: row anchor 기반 이산 표현 (Ultra-Fast-Lane-Detection 계열)
- **Bezier head**: 4 control point Bezier 기반 연속 표현
- **Routing head**: 각 `(lane, row)`에서 Anchor와 Bezier를 섞는 gate
- 학습은 **Phase 1–4 (P1–P4)** 로 나뉘며, 각 Phase는 서로 다른 역할을 가진다.

---

## Model Overview

모델 전체 구조는 아래와 같다.

- 하나의 shared backbone feature 위에
  - Anchor head (anchor 포맷 예측)
  - Bezier head (연속 곡선 예측)
  - Routing head (두 표현을 per-lane, per-row로 조합)
- 최종 출력은 항상 **anchor 포맷(x at row anchors)** 이며, Bezier는 보조 expert + teacher 역할을 한다.

```text
Backbone (ResNet18 + FPN)
        │
        ├── Anchor head  → x_anchor (B × L × R), exist_logit (B × L × R)
        ├── Bezier head  → ctrl_points (B × L × 4 × 2)
        └── Routing head → gate (B × L × R)
```

![Dual-Head Lane Architecture](assets/dual_head_lane_arch.png)

---

## 0. 전체 컨셉

### 두 표현의 역할

- **Anchor 표현 (Anchor head)**
  - 고정된 row 위치에서의 x 좌표들로 차선을 이산적으로 표현
  - CULane 같은 **anchor 기반 평가 포맷**과 직접 호환
  - “이 y에서 x가 몇 픽셀인지” 하는 **local 좌표**를 정확히 맞추는 데 강함

- **Bezier 표현 (Bezier head)**
  - 각 lane을 **4개의 control point**로 표현하는 3차 Bezier 곡선
  - occlusion, 가림, 차선 끊김, 큰 곡률 등에서
    - 보이는 부분만이 아니라 **전체 곡선 형태**를 유지하기 쉬움

- **Routing head (gate)**
  - 각 `(lane, row)` 위치마다
    - “여기서는 Anchor 값이 더 낫다”
    - “여기서는 Bezier에서 유도된 값이 더 낫다”
    를 결정해서 최종 `x_mix`를 만든다.

결론적으로, 이 구조는 **“직선 vs 곡선”이 아니라, Anchor / Bezier 두 표현의 역할 분담 + 동적 조합**에 가깝다.

---

## 1. 모델 구조

### 1.1 Backbone

- ResNet18 + FPN 계열 backbone
- 출력 feature:
  - `feat ∈ ℝ^{B×C×H'×W'}`
- 이 feature를 세 갈래로 사용:
  - Anchor head
  - Bezier head
  - Routing head

---

### 1.2 Anchor head (Anchor 기반 표현)

- 모듈: `neck_straight` → `head_straight`
- 출력:
  - `x_anchor ∈ ℝ^{B×L×R}`
    - `L`: 최대 차선 수
    - `R`: row anchor 개수
    - `(b, l, r)`에서 row `r` 위치의 x 픽셀 좌표
  - `exist_logit ∈ ℝ^{B×L×R}`: 각 `(lane, row)`에서 lane 존재 여부

특징:

- Ultra-Fast-Lane-Detection 스타일 **row-anchor 표현**
- CULane 평가 포맷과 직접 맞닿아 있음
- local position error에 민감한 **anchor 기반 평가**에서 강점

---

### 1.3 Bezier head (Bezier 기반 연속 표현)

- 모듈: `neck_curve` → `head_curve`
- 출력:
  - `ctrl_points ∈ ℝ^{B×L×4×2}`
    - 각 lane 마다 `(x, y)` control point 4개 (`P0..P3`)

Bezier 곡선:

- 각 lane l에 대해 control points `P0..P3` 로 3차 Bezier 곡선 생성
- 구현에서는:
  - `sample_bezier(ctrl_points, T) → (B,L,T,2)`
  - T개의 샘플 포인트로 polyline을 만든 후 loss/시각화/정렬에 사용

특징:

- 하나의 연속 곡선으로 lane 전체를 본다.
- **긴 거리 / 큰 곡률 / 가림 / 끊김** 구간에서 전체 패턴을 유지하기 유리하다.

---

### 1.4 Routing head (Anchor / Bezier 조합 gate)

- 입력: backbone feature `feat ∈ ℝ^{B×C×H'×W'}`
- 구조 (요약):
  - conv 블록을 거쳐 `(B,L,H',W')` 형태로 만든 뒤
  - `adaptive_avg_pool2d(..., (R,1)) → (B,L,R)`
  - `sigmoid` → `gate ∈ (0,1)^{B×L×R}`

Routing은 다음처럼 동작한다:

1. Bezier control로부터 row-anchor 위치의 x 좌표를 얻음:
   - `x_bezier_row ∈ ℝ^{B×L×R}`

2. Anchor / Bezier를 gate로 섞어 최종 x 생성:

   - `x_mix = (1 - gate) * x_anchor + gate * x_bezier_row`
   - `x_mix`가 최종 Anchor 포맷 prediction이 되어 **평가에 사용**된다.

---

## 2. GT 표현 (Anchor / Polyline)

### 2.1 Polyline GT (Bezier용)

- CULane `.lines.txt` 에서 lane별 polyline (x, y)들을 읽는다.
- 각 lane을 T개 포인트로 resample:
  - `gt_polyline ∈ ℝ^{B×L×T×2}`
- lane 존재 마스크:
  - `lane_mask ∈ {0,1}^{B×L}`

Bezier head 학습(`curve_loss`)에 직접 사용된다.

---

### 2.2 Anchor GT (Anchor / Routing용)

- 미리 정의된 row anchor:
  - `row_anchor_ys ∈ ℝ^{R}`

- 각 lane polyline에서, 모든 row y에 대해 x를 선형 보간해서 얻는다:

  - `gt_anchor['x'] ∈ ℝ^{B×L×R}`: 해당 row에서의 x
  - `gt_anchor['mask'] ∈ {0,1}^{B×L×R}`: 해당 위치의 GT가 유효한지
  - `gt_anchor['exist'] ∈ {0,1}^{B×L}`: lane 존재 여부 (현재 loss에서는 주로 `mask`를 사용)

Anchor head (`anchor_loss`) 및 Routing head (`routing_loss`) 에 사용된다.

---

## 3. Loss 설계

### 3.1 Bezier 손실: `curve_loss`

목적: Bezier head가 **연속적인 lane 전체 형태**를 잘 맞추도록.

1. pred Bezier → T개 샘플 포인트:
   - `pred_pts = sample_bezier(ctrl_points, T) ∈ ℝ^{B×L×T×2}`

2. Polyline GT (`gt_polyline`)와 L1 loss:
   - lane 존재 마스크 `lane_mask`로 valid lane만 평균

직관적으로, Bezier 곡선이 실제 lane polyline과 최대한 가깝게 지나가도록 만든다.

---

### 3.2 Anchor 손실: `anchor_loss` (straight_loss)

목적: Anchor head가 **anchor GT** 기준으로 정확한 x를 맞추도록.

구성:

1. 위치 L1:

   - `L_anchor_pos = mean_L1( x_anchor, gt_anchor['x'], mask = gt_anchor['mask'] )`

2. per-(lane,row) 존재 BCE:

   - `exist_logit ∈ ℝ^{B×L×R}`
   - `gt_exist = (gt_anchor['mask'] > 0.5) ∈ {0,1}^{B×L×R}`
   - `L_anchor_exist = BCEWithLogits( exist_logit, gt_exist )` (mask를 통해 평균)

3. 최종:

   - `L_anchor = L_anchor_pos + λ_exist * L_anchor_exist`

---

### 3.3 Anchor–Bezier 정렬 손실: `consistency_loss` (Phase3)

목적: 같은 row index에서 Anchor / Bezier 표현이 **완전히 따로 놀지 않게** 정렬.

구현 관점에서:

1. Bezier control을 `num_samples = R` 로 균일 샘플:
   - `pts = sample_bezier(ctrl_points_detached, R) ∈ ℝ^{B×L×R×2}`
   - 여기서 `ctrl_points_detached` 는 gradient를 끊은 버전 (`detach()`) → **consistency 항에서는 Bezier가 teacher 역할**

2. 각 샘플의 x만 추출해서:
   - `x_bezier_row_R = pts[..., 0] ∈ ℝ^{B×L×R}`

3. Anchor vs Bezier L1:

   - `L_cons = mean_L1( x_anchor, x_bezier_row_R, mask = gt_anchor['mask'] )`

Phase3 전체 loss는 대략:

- `L_phase3 = L_anchor + λ_curve * L_curve + λ_cons * L_cons`

중요한 점:

- **Bezier head는 Phase3에서도 `curve_loss`로는 계속 업데이트**되지만,
- `consistency_loss` 에서는 `detach()`된 control point를 사용해서 **teacher처럼만 쓰인다.**

---

### 3.4 Routing 손실: `routing_loss` (Phase4)

목적: 각 `(lane, row)`에서 **어디서 Bezier를 더 쓸지, 어디서 Anchor를 더 쓸지** 학습.

1. Bezier → row anchor:
   - `x_bezier_row = bezier_x_on_rows(ctrl_points, row_anchor_ys, lane_mask)`
   - Bezier polyline을 촘촘히 샘플한 뒤, 각 row y에 대해 가장 가까운 y 위치를 찾아서 x를 가져오는 방식
   - shape: `(B, L, R)`

2. gate로 mix:

   - `x_mix = (1 - gate) * x_anchor + gate * x_bezier_row`

3. GT 기준 mix L1:

   - `L_mix = mean_L1( x_mix, gt_anchor['x'], mask = gt_anchor['mask'] )`

4. gate soft target (어디서 Bezier가 나은지):

   - `e_anchor = |x_anchor - gt_x|`
   - `e_bezier = |x_bezier_row - gt_x|`
   - Bezier가 더 나을수록 gate↑:
     - `g_target = sigmoid( (e_anchor - e_bezier) / τ )` (detach로 gradient 막음)

5. gate BCE:

   - `L_gate = mean_BCE( gate, g_target, mask = gt_anchor['mask'] )`

6. 최종 Routing loss:

   - `L_routing = L_mix + α_gate * L_gate`

Phase4에서는 **Routing head만 학습**, Backbone / Anchor head / Bezier head는 모두 freeze 한다.

---

## 4. 학습 Phase 정리 (P1–P4)

### 4.1 Phase별 의미

| Phase | 내부 이름 (`--phase`) | 학습 대상                              | Freeze                             | 주요 Loss                                                 | 역할 |
|:-----:|------------------------|----------------------------------------|------------------------------------|-----------------------------------------------------------|------|
| P1    | `curve_only`           | Backbone + Bezier head                | Anchor / Routing                   | `curve_loss`                                              | Bezier **expert (teacher)** 확보 |
| P2    | `straight_only`        | Backbone + Anchor head                | Bezier / Routing                   | `anchor_loss`                                             | Anchor **expert** 확보 |
| P3    | `joint`                | Backbone + Anchor head + (Bezier 일부) | Routing                             | `anchor_loss + λ_curve * curve_loss + λ_cons * cons`     | Anchor를 Bezier teacher 쪽으로 정렬 |
| P4    | `route`                | Routing head                           | Backbone + Anchor + Bezier         | `routing_loss = L_mix + α_gate * L_gate`                 | Anchor / Bezier 조합 최적화 |

- P3에서:
  - Bezier는 `curve_loss` 기준으로 여전히 업데이트되지만,
  - `consistency_loss` 에서는 gradient를 받지 않는 teacher 역할을 한다.

---

### 4.2 Checkpoint 예시 (P1–P4 매핑)

학습이 끝난 checkpoint 예시는 다음과 같이 매핑할 수 있다.

- **P1 (Bezier)**  
  `runs_dual/phase1_curve/dual_curve_only_epoch_10.pth`
- **P2 (Anchor)**  
  `runs_dual/phase2_straight/dual_straight_only_epoch_10.pth`
- **P3 (Anchor + Bezier 정렬)**  
  `runs_dual/phase3_joint/dual_joint_epoch_5.pth`
- **P4 (Routing mix)**  
  `runs_dual/phase4_route/dual_route_epoch_5.pth`

평가/시각화/표에서는 항상:

- `P1 = Bezier`
- `P2 = Anchor`
- `P3 = Anchor (joint)`
- `P4 = Mix (Routing)`

이 순서로 고정해서 사용한다.

---

## 5. 평가 관점에서 P1–P4

### 5.1 P1–P4의 의미 (Evaluation View)

- **P1 (Bezier)**  
  - Bezier control points만 사용해서, row anchor 위치로 투영한 결과.
  - “Bezier 표현 자체가 anchor 포맷에서도 어느 정도까지 성능이 나오는가”를 보는 baseline.

- **P2 (Anchor)**  
  - 순수 Anchor expert.
  - CULane anchor 포맷 기준으로 가장 직접적인 모델.

- **P3 (Anchor, joint)**  
  - Anchor를 Bezier teacher와 consistency loss로 정렬한 버전.
  - Anchor 성능은 유지하면서, Bezier 곡선과의 연속성/형태도 어느 정도 따라가게 만든 상태.

- **P4 (Mix, Routing)**  
  - Anchor + Bezier를 Routing head로 per-row 조합한 최종 모델.
  - occlusion, 큰 곡률 등에서는 Bezier 쪽을 더 쓰고,
    노멀한 구간에서는 Anchor 쪽을 더 쓰는 방향으로 학습된다.

표/그래프에서는 보통 다음 순서로 열을 나열한다:

- `P1 Bezier`, `P2 Anchor`, `P3 Anchor`, `P4 Mix`

---

## 6. 실행 예시

### 6.1 Phase별 학습 예시

실제 옵션/인자는 레포 코드와 맞춰야 하지만, 흐름은 아래와 같다.

```bash
# Phase 1: Bezier teacher 학습
python train_dual.py \
    --phase curve_only \
    --data_root /path/to/CULane \
    --list_train /path/to/CULane/list/train.txt \
    --list_val /path/to/CULane/list/val.txt \
    --save_dir ./runs_dual/phase1_curve \
    --num_epochs 10

# Phase 2: Anchor expert 학습
python train_dual.py \
    --phase straight_only \
    --data_root /path/to/CULane \
    --list_train /path/to/CULane/list/train.txt \
    --list_val /path/to/CULane/list/val.txt \
    --save_dir ./runs_dual/phase2_straight \
    --resume ./runs_dual/phase1_curve/dual_curve_only_epoch_10.pth \
    --num_epochs 10

# Phase 3: Anchor–Bezier joint (consistency)
python train_dual.py \
    --phase joint \
    --data_root /path/to/CULane \
    --list_train /path/to/CULane/list/train.txt \
    --list_val /path/to/CULane/list/val.txt \
    --save_dir ./runs_dual/phase3_joint \
    --resume ./runs_dual/phase2_straight/dual_straight_only_epoch_10.pth \
    --num_epochs 5

# Phase 4: Routing head 학습
python train_dual.py \
    --phase route \
    --data_root /path/to/CULane \
    --list_train /path/to/CULane/list/train.txt \
    --list_val /path/to/CULane/list/val.txt \
    --save_dir ./runs_dual/phase4_route \
    --resume ./runs_dual/phase3_joint/dual_joint_epoch_5.pth \
    --num_epochs 5
```

---

### 6.2 P1–P4를 한 번에 시각화 (같은 이미지에서 비교)

`visualize_all_phases.py` 스크립트를 사용해서, 동일 이미지에 P1–P4 + GT를 모두 올려서 비교할 수 있다.

```bash
python visualize_all_phases.py \
    --data_root /path/to/CULane \
    --list_val /path/to/CULane/list/test.txt \
    --ckpt_phase1 ./runs_dual/phase1_curve/dual_curve_only_epoch_10.pth \
    --ckpt_phase2 ./runs_dual/phase2_straight/dual_straight_only_epoch_10.pth \
    --ckpt_phase3 ./runs_dual/phase3_joint/dual_joint_epoch_5.pth \
    --ckpt_phase4 ./runs_dual/phase4_route/dual_route_epoch_5.pth \
    --out_dir ./vis_all_phases \
    --num_vis 50 \
    --num_workers 0
```

시각화 규칙 예시 (코드와 맞춰 조정 가능):

- GT: 빨간색 점 (red dots)
- P1 (Bezier): 초록색 선 또는 polyline
- P2 (Anchor): 예) 파란색 삼각형 marker
- P3 (Anchor joint): 예) 주황색 네모 marker
- P4 (Mix): 예) 보라색 별 marker
- 그림 안 legend로 `GT / P1 / P2 / P3 / P4` 표시

---

### 6.3 Polyline 기반 평가 스크립트

`eval_polyline_metrics_all.py` 를 사용해서, P1–P4에 대해 polyline L1, smoothness 등 지표를 한 번에 계산할 수 있다.

```bash
python eval_polyline_metrics_all.py \
    --data_root /path/to/CULane \
    --list_val /path/to/CULane/list/test.txt \
    --ckpt_phase1 ./runs_dual/phase1_curve/dual_curve_only_epoch_10.pth \
    --ckpt_phase2 ./runs_dual/phase2_straight/dual_straight_only_epoch_10.pth \
    --ckpt_phase3 ./runs_dual/phase3_joint/dual_joint_epoch_5.pth \
    --ckpt_phase4 ./runs_dual/phase4_route/dual_route_epoch_5.pth
```

---

## 7. Repository Structure (예시)

```text
.
├── README.md
├── train_dual.py
├── dataset_dual.py
├── losses_dual.py
├── bezier_utils.py
├── eval_polyline_metrics_all.py
├── visualize_all_phases.py
├── assets/
│   └── dual_head_lane_arch.png
└── scripts/
    ├── train_phase1_curve.sh
    ├── train_phase2_straight.sh
    ├── train_phase3_joint.sh
    ├── train_phase4_route.sh
    ├── eval_polyline_metrics.sh
    └── viz_all_phases.sh
```

---

## 8. Polyline / Smoothness Metrics

### 8.1 Polyline L1 vs GT (per-lane)

```text
[Polyline L1 vs GT] (per-lane)
P1 Bezier : mean = 105.525, std = 66.348, num_lanes = 104886
P2 Anchor : mean =  50.278, std = 54.240, num_lanes = 104886
P3 Anchor : mean =  46.778, std = 52.641, num_lanes = 104886
P4 Mix    : mean =  46.413, std = 52.556, num_lanes = 104886
```

표 형식으로 정리하면 다음과 같다.

| Phase | Head   | mean L1 | std L1  | #lanes  |
|:-----:|--------|--------:|--------:|--------:|
| P1    | Bezier | 105.525 | 66.348  | 104886  |
| P2    | Anchor |  50.278 | 54.240  | 104886  |
| P3    | Anchor |  46.778 | 52.641  | 104886  |
| P4    | Mix    |  46.413 | 52.556  | 104886  |

해석:

- **P1 (Bezier)**  
  - anchor 포맷으로 투영했을 때 L1 오차가 크게 나온다.  
  - Bezier는 **연속적인 곡선 형태**를 잘 잡지만, 일정한 row anchor grid에서의 local x 오차 기준으로는 손해를 본다.
- **P2 (Anchor)**  
  - 순수 anchor expert. P1보다 L1이 크게 줄어든다.
- **P3 (Anchor joint)**  
  - P2보다 mean L1이 더 낮다.  
  - Anchor 표현이 Bezier teacher와 정렬되면서, anchor 포맷 기준에서도 약간 더 좋아진다.
- **P4 (Mix, Routing)**  
  - 네 모델 중 **가장 낮은 mean L1**을 기록한다.  
  - Routing head가 “어디서는 Anchor, 어디서는 Bezier” 를 잘 선택해서  
    anchor GT 기준 오차를 최소화하고 있다는 것을 보여준다.  
  - P3와 비교하면 차이는 크지 않지만, **조합 전략이 순수 Anchor보다 이득**이 있음을 수치로 확인할 수 있다.

요약하면:

- anchor 기준 L1 에서는 **P4 (Mix) > P3 ≥ P2 ≫ P1** 순으로 좋다.
- Bezier 표현은 단독으로 쓸 때 anchor 포맷과는 거리가 있지만,  
  Routing으로 잘 섞으면 anchor 포맷에서도 best 성능을 낼 수 있다는 걸 보여준다.

---

### 8.2 Smoothness (Δ² x, per-lane)

```text
[Smoothness (Δ² x, per-lane)]
GT        : mean =  1.023387
P1 Bezier : mean =  0.964085
P2 Anchor : mean = 16.132438
P3 Anchor : mean = 15.972920
P4 Mix    : mean = 15.962773
```

표 형식:

| Model   | mean Δ²x  |
|---------|----------:|
| GT      |  1.023387 |
| P1 Bezier | 0.964085 |
| P2 Anchor | 16.132438 |
| P3 Anchor | 15.972920 |
| P4 Mix    | 15.962773 |

해석:

- **GT vs P1 (Bezier)**  
  - GT: 1.02, P1: 0.96 으로 **거의 같은 수준의 smoothness**.  
  - Bezier 기반 표현은 **곡선의 곡률 변화(두 번째 차분)** 관점에서 GT와 매우 유사한 부드러움을 갖는다.
- **P2 / P3 / P4 (Anchor / Mix)**  
  - smoothness 값이 ~16 근처로, **GT / Bezier보다 약 15배 정도 거칠다.**  
  - row anchor마다 독립적으로 x를 예측하다 보니,  
    곡선 관점에서 보면 **“지그재그 / 끊긴 듯한”** polyline이 된다.
  - P2 → P3 → P4 로 갈수록 smoothness는 아주 조금씩만 좋아지고,  
    본질적으로는 **anchor 기반 표현의 한계**가 유지되는 것을 보여준다.

이 두 지표를 합쳐보면:

- **정확도(Polyline L1)** 관점에서는  
  - P1(Bezier)은 약하고, P2/P3/P4(Anchor / Mix)가 강하다.
- **연속성 / 곡률(Smoothness)** 관점에서는  
  - GT ≈ P1(Bezier)이 가장 자연스럽고,  
  - P2/P3/P4는 anchor grid 때문에 부드럽지 않다.

즉,

- Bezier head는 **형태(smoothness)** 는 좋지만, anchor 포맷 기준 local 정확도는 떨어지는 expert.
- Anchor head / Mix는 **local 정확도(L1)** 는 좋지만, 곡선 smoothness는 나쁜 expert.
- Routing head는 이 둘을 섞어서:
  - **anchor 기준 L1은 최대로 낮추면서** (P4가 best),
  - 일부 row에서는 Bezier를 더 써서 **형태적인 이득을 부분적으로 가져가는 구조**다.

---

## 9. 요약

- 이 레포는 **Anchor 기반 이산 표현**과 **Bezier 기반 연속 표현**을 동시에 사용하고,
- **Phase 1–4 (P1–P4)** 에 걸쳐
  - Bezier expert (teacher) 학습
  - Anchor expert 학습
  - Anchor–Bezier joint 정렬
  - Routing head로 두 표현 조합
- 까지 점진적으로 모델을 만드는 구조다.

정량 지표 관점에서:

- Polyline L1:
  - `P4 Mix` 가 anchor-space 정확도에서 가장 좋다.
- Smoothness (Δ² x):
  - `GT` 와 `P1 Bezier` 가 가장 부드러운 곡선을 가진다.
  - Anchor / Mix 기반 표현은 정확도는 좋지만, 곡선 관점에서는 거칠다.

이 결과는:

- **Anchor vs Bezier가 서로 다른 관점(정확도 vs 형태)의 expert** 라는 점,
- **Routing head가 둘을 per-row로 조합해 anchor-space 성능을 끌어올린다는 점**

을 뒷받침하는 실험적 근거로 해석할 수 있다.
