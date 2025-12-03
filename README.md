# anchor-bezier-lanenet 
### Anchor × Bezier × Routing (Mixture-of-Experts Style)

이 프로젝트는 **Anchor 기반 이산 표현**과  
**Bezier 기반 연속 표현**을 동시에 학습하고,  
마지막에 **Routing head**를 통해 두 표현을 동적으로 조합하는  
**Dual-Representation Lane Detection** 방법을 구현한다.

- 표현 1: **Anchor head** – row-anchor 기준 x 좌표 (CULane 스타일)
- 표현 2: **Bezier head** – control point 4개로 곡선을 연속적으로 표현
- 표현 3: **Routing head** – (lane, row) 단위로 Anchor vs Bezier 비중을 결정

---

# 0. 전체 컨셉

본 방식은 “직선 vs 곡선”이 아니라,

> **같은 차선 정보를 두 가지 상호 보완적인 표현으로 잡고,  
> 상황에 따라 어떤 표현을 더 신뢰할지 학습하는 구조이다.**

- **Anchor 표현**
  - 고정된 row-anchor에서 x 픽셀 좌표를 이산적 형태로 제공
  - CULane 평가 프로토콜과 직접적으로 호환
  - local alignment, 수직 샘플 기반 평가에서 강함

- **Bezier 표현**
  - 4개의 control point로 연속적 곡선을 정의
  - 곡률 변화, 차선 가림(occlusion), 끊김 구간 등에서 안정적
  - 전역적 형태를 표현하는 능력이 anchor보다 우수

- **Routing head**
  - 각 lane, 각 row마다 **“Anchor를 신뢰할지 Bezier를 신뢰할지”** gate 값을 학습
  - 최종 예측:
    ```
    x_mix = (1 - gate) * x_anchor + gate * x_bezier_row
    ```

---

# 1. 모델 구조

> ※ 아래 이미지처럼 모델 구조도를 만들고 넣으면 된다.  
> (이미지는 나중에 넣으세요)


## 1.1 Backbone

- ResNet18 + FPN 계열 feature extractor
- 출력 feature: `B × C × H' × W'`
- 동일 feature를 Anchor/Bezier/Routing 세 곳에 전달

## 1.2 Anchor Head

- 입력: backbone feature
- 출력:
  - `x_anchor ∈ B × L × R`
  - `exist_logit ∈ B × L`
- 특징:
  - row-anchor 기반 이산 lane 표현
  - CULane와 같은 anchor 기반 평가 구조에 최적화

## 1.3 Bezier Head

- 입력: backbone feature
- 출력:
  - `ctrl_points ∈ B × L × 4 × 2`
- 특징:
  - control point 4개로 곡선을 전역적으로 표현
  - Bezier 곡선은 T개 점으로 샘플링해 GT와 비교

## 1.4 Routing Head (gate)

- 입력: backbone feature
- 출력:  
  `gate ∈ B × L × R`
- 역할:
  - Anchor 기반 x vs Bezier 기반 x를 **per-row** 단위로 혼합(weighting)

- 최종:
  - &&x_mix[b, l, r] = (1 - gate[b,l,r]) * x_anchor[b,l,r] + gate[b,l,r] * x_bezier_row[b,l,r]&&


---

# 2. GT 표현 방식 (Anchor GT / Polyline GT)

## 2.1 Polyline GT (Bezier용)

- `.lines.txt`에서 lane polyline (x,y) 좌표들을 읽음
- 일정 개수 T로 리샘플해:
  &&gt_polyline ∈ B × L × T × 2
lane_mask ∈ B × L&&


## 2.2 Anchor GT (Anchor head용)

- 고정된 row-anchor 위치 `R` 개에 대해  
polyline을 선형 보간하여 anchor GT 생성:

gt_anchor['x'] ∈ B × L × R
gt_anchor['mask'] ∈ B × L × R
gt_anchor['exist']∈ B × L
