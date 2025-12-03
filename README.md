# Dual-Head Lane Detection with Anchor / Bezier / Routing

이 레포는 **같은 차선 정보를 두 가지 표현(Anchor / Bezier)으로 잡고, 상황에 따라 어느 쪽을 더 신뢰할지 라우팅하는 구조**를 구현한다.

- **Anchor head**: row anchor 기반 이산 표현 (Ultra-Fast-Lane-Detection 계열)
- **Bezier head**: 4 control point Bezier 기반 연속 표현
- **Routing head**: 각 `(lane, row)`에서 Anchor vs Bezier를 섞는 gate
- 학습은 **Phase 1–4 (P1–P4)** 로 나뉘며, 각 Phase는 서로 다른 역할을 가진다.

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
  - 를 결정해서 최종 `x_mix`를 만든다.

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
  - `exist_logit ∈ ℝ^{B×L}`: 각 lane 존재 여부

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
  - `exist_logits ∈ ℝ^{B×L}`

Bezier 곡선:

- 각 lane l에 대해 control points `P0..P3` 로 3차 Bezier 곡선 생성
- 구현에서는:
  - `sample_bezier(ctrl_points, T) → (B,L,T,2)`
  - T개의 샘플 포인트로 polyline을 만든 후 loss/시각화에 사용

특징:

- 하나의 연속 곡선으로 lane 전체를 본다.
- **긴 거리 / 큰 곡률 / 가림 / 끊김** 구간에서 전체 패턴을 유지하기 유리하다.

---

### 1.4 Routing head (Anchor / Bezier 조합 gate)

- 입력: backbone feature `feat ∈ ℝ^{B×C×H'×W'}`
- 구조 (요약):
  - conv 2개로 `(B,L,H',W')` 형태로 만든 뒤
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
  - `gt_anchor['exist'] ∈ {0,1}^{B×L}`: lane 존재 여부

Anchor head (`anchor_loss`) 및 Routing head (`routing_loss`) 에 사용된다.

---

## 3. Loss 설계

### 3.1 Bezier 손실: `curve_loss`

목적: Bezier head가 **연속적인 lane 전체 형태**를 잘 맞추도록.

- pred Bezier → T개 샘플 포인트:
  - `pred_pts = sample_bezier(ctrl_points, T) ∈ ℝ^{B×L×T×2}`
- Polyline GT (`gt_polyline`)와 L1 loss:
  - lane 존재 마스크 `lane_mask`로 평균

직관적으로, Bezier 곡선이 실제 lane polyline과 최대한 가깝게 지나가도록 만든다.

---

### 3.2 Anchor 손실: `anchor_loss`

목적: Anchor head가 **anchor GT** 기준으로 정확한 x를 맞추도록.

구성:

1. 위치 L1:

   - `L_anchor_pos = mean_L1( x_anchor, gt_anchor['x'], mask = gt_anchor['mask'] )`

2. lane 존재 BCE:

   - `L_anchor_exist = BCEWithLogits( exist_logit, gt_anchor['exist'] )`

3. 최종:

   - `L_anchor = L_anchor_pos + λ_exist * L_anchor_exist`

---

### 3.3 Anchor–Bezier 정렬 손실: `consistency_loss` (Phase3)

목적: 같은 row에서 Anchor / Bezier 표현이 **완전히 따로 놀지 않게** 정렬.

1. Bezier 샘플 포인트로부터 각 row anchor y와 가장 가까운 y의 x를 가져와:
   - `x_bezier_row ∈ ℝ^{B×L×R}` (teacher 역할)

2. Anchor vs Bezier L1:

   - `L_cons = mean_L1( x_anchor, x_bezier_row, mask = gt_anchor['mask'] )`

Phase3 전체 loss:

- `L_phase3 = L_anchor + λ_curve * L_curve + λ_cons * L_cons`
- 이때 **Bezier head는 freeze (teacher)** 이고,
- Anchor head + backbone이 Bezier teacher 쪽으로 조금 끌려가는 구조.

---

### 3.4 Routing 손실: `routing_loss` (Phase4)

목적: 각 `(lane, row)`에서 **어디서 Bezier를 더 쓸지, 어디서 Anchor를 더 쓸지** 학습.

1. Bezier → row anchor:
   - `x_bezier_row = bezier_x_on_rows(ctrl_points, row_ys, lane_mask)`
   - shape: `(B, L, R)`

2. gate로 mix:

   - `x_mix = (1 - gate) * x_anchor + gate * x_bezier_row`

3. GT 기준 mix L1:

   - `L_mix = mean_L1( x_mix, gt_anchor['x'], mask = gt_anchor['mask'] )`

4. gate soft target (어디서 Bezier가 나은지):

   - `e_anchor = |x_anchor - gt_x|`
   - `e_bezier = |x_bezier_row - gt_x|`
   - Bezier가 더 나을수록 gate↑:
     - `g_target = sigmoid( (e_anchor - e_bezier) / τ )`

5. gate BCE:

   - `L_gate = mean_BCE( gate, g_target, mask = gt_anchor['mask'] )`

6. 최종 Routing loss:

   - `L_routing = L_mix + α_gate * L_gate`

Phase4에서는 **Routing head만 학습**, Anchor / Bezier / backbone은 모두 freeze 한다.

---

## 4. 학습 Phase 정리 (P1–P4)

### 4.1 Phase별 의미

| Phase | 내부 이름(예시)         | 학습 대상                           | Freeze                    | 주요 Loss                                             | 역할 |
|-------|-------------------------|-------------------------------------|---------------------------|-------------------------------------------------------|------|
| P1    | `phase1_curve_only`     | Backbone + Bezier head             | Anchor / Routing          | `curve_loss`                                          | Bezier **teacher** 확보 |
| P2    | `phase2_straight_only`  | Backbone + Anchor head             | Bezier / Routing          | `anchor_loss`                                         | Anchor **expert** 확보 |
| P3    | `phase3_joint`          | Backbone + Anchor head             | Bezier (teacher) / Routing | `anchor_loss + λ_curve * curve_loss + λ_cons * cons` | Anchor를 Bezier teacher 쪽으로 정렬 |
| P4    | `phase4_route`          | Routing head                        | Backbone + Anchor + Bezier | `routing_loss = L_mix + α_gate * L_gate`             | Anchor / Bezier 조합 최적화 |

---

### 4.2 Checkpoint 예시 (P1–P4 매핑)

레포에서 사용하는 ckpt 네이밍 예시:

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
- `P3 = Anchor (joint 정렬)`
- `P4 = Mix (Routing)`

이 순서로 고정해서 사용한다.

---

## 5. 평가 관점에서 P1–P4

### 5.1 P1–P4의 의미 (Evaluation View)

- **P1 (Bezier)**  
  - Bezier control points만 사용해서, row anchor 위치로 투영한 결과.
  - “Bezier 표현 자체가 어느 정도까지 anchor 포맷에서도 성능이 나오는가”를 보는 baseline.

- **P2 (Anchor)**  
  - 순수 Anchor expert.
  - CULane anchor 포맷 기준으로 가장 직접적인 모델.

- **P3 (Anchor, joint)**  
  - Anchor를 Bezier teacher와 consistency loss로 정렬한 버전.
  - Anchor 성능은 유지하면서, Bezier 곡선과의 연속성/형태도 어느 정도 따라가게 만든 상태.

- **P4 (Mix, Routing)**  
  - P2/P3의 Anchor + P1의 Bezier를 Routing head로 per-row 조합한 최종 모델.
  - occlusion, 큰 곡률 등에서는 Bezier 쪽을 더 쓰고,
    노멀한 구간에서는 Anchor 쪽을 더 쓰는 방향을 학습.

표/그래프에서는 보통 다음 순서로 열을 나열한다:

- `P1 Bezier`, `P2 Anchor`, `P3 Anchor`, `P4 Mix`

---

## 6. 실행 예시

### 6.1 Phase별 학습 예시 (pseudo command)

실제 옵션/인자는 레포 코드와 맞춰야 하지만, 흐름은 아래와 같다.

```bash
# Phase 1: Bezier teacher 학습
python train.py   --phase curve_only   --exp_name phase1_curve   ...

# Phase 2: Anchor expert 학습
python train.py   --phase straight_only   --resume runs_dual/phase1_curve/dual_curve_only_epoch_10.pth   --exp_name phase2_straight   ...

# Phase 3: Anchor–Bezier joint (consistency)
python train.py   --phase joint   --resume runs_dual/phase2_straight/dual_straight_only_epoch_10.pth   --exp_name phase3_joint   ...

# Phase 4: Routing head 학습
python train.py   --phase route   --resume runs_dual/phase3_joint/dual_joint_epoch_5.pth   --exp_name phase4_route   ...
```

---

### 6.2 P1–P4를 한 번에 시각화 (같은 이미지에서 비교)

`visualize_all_phases.py` 같은 스크립트를 사용해서, 동일 이미지에 P1–P4 + GT를 모두 올려서 비교한다.

```bash
python visualize_all_phases.py   --data_root /path/to/CULane   --list_val /path/to/CULane/list/test.txt   --ckpt_phase1 runs_dual/phase1_curve/dual_curve_only_epoch_10.pth   --ckpt_phase2 runs_dual/phase2_straight/dual_straight_only_epoch_10.pth   --ckpt_phase3 runs_dual/phase3_joint/dual_joint_epoch_5.pth   --ckpt_phase4 runs_dual/phase4_route/dual_route_epoch_5.pth   --out_dir vis_all_phases   --num_vis 50
```

시각화 규칙 예시 (레포에 맞게 조정):

- GT: 빨간색 점 (red dots)
- P1 (Bezier): 초록색 선 또는 polyline
- P2 (Anchor): 예) 파란색 삼각형 marker
- P3 (Anchor joint): 예) 주황색 네모 marker
- P4 (Mix): 예) 보라색 별 marker
- 오른쪽 위에 legend로 `GT / P1 / P2 / P3 / P4` 표시

---

### 6.3 P1–P4 중 “차이가 큰 이미지”만 골라내는 비교 시각화

`viz_compare_all_phases.py` 같은 스크립트로, L1 error 차이가 가장 큰 이미지들만 뽑아서 P1–P4를 한 번에 그려볼 수 있다.

```bash
python viz_compare_all_phases.py   --data_root /path/to/CULane   --list_val /path/to/CULane/list/test.txt   --ckpt_phase1 runs_dual/phase1_curve/dual_curve_only_epoch_10.pth   --ckpt_phase2 runs_dual/phase2_straight/dual_straight_only_epoch_10.pth   --ckpt_phase3 runs_dual/phase3_joint/dual_joint_epoch_5.pth   --ckpt_phase4 runs_dual/phase4_route/dual_route_epoch_5.pth   --out_dir vis_compare_all_phases   --num_vis 50   --num_workers 0
```

이 스크립트에서는, 예를 들어:

- P1이 GT에 더 가까운 케이스
- P2, P3, P4가 점점 개선되는 케이스
- Routing(P4)이 특히 이득을 보는 케이스

같은 것들을 따로 골라서 저장할 수 있다.

---

## 7. 요약

- 이 레포는 **Anchor 기반 이산 표현**과 **Bezier 기반 연속 표현**을 동시에 사용하고,
- **Phase 1–4 (P1–P4)** 에 걸쳐
  - Bezier expert (teacher) 학습
  - Anchor expert 학습
  - Anchor–Bezier 정렬
  - Routing head로 두 표현 조합
- 까지 점진적으로 모델을 만드는 구조다.

평가 및 시각화에서는:

- `P1 = Bezier`, `P2 = Anchor`, `P3 = Anchor(joint)`, `P4 = Mix(route)`
- 이 **고정된 순서**를 기준으로 결과를 해석하면 된다.
