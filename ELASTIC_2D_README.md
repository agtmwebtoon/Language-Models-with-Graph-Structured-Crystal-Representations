# 2D Elastic Tensor Regression with Symmetry Awareness

## 개요

Robocrystallographer 텍스트로부터 2D 탄성 텐서를 예측하는 모델입니다.

**핵심 아이디어**:
- 항상 일반형(triclinic) 6-DOF로 예측
- 여러 symmetry 제약으로의 투영을 혼합
- Robo-based prior + learnable correction

## 모델 구조

```
Text (Robo) → CLIP Encoder → Dual Head
                              ├─ Tensor Head → pred6 (6 components)
                              └─ Symmetry Head → α (4 mixture weights)

pred6: [C11, C22, C12, C66, C16, C26] (always 6-DOF)
α: [p_tri, p_rect, p_tetra, p_hexa]
   = softmax(z + β·log p_prior)
```

## 사용 방법

### 1. Dataset에 `group` 컬럼 추가

JSONL 파일에 Robo crystal system 정보 추가:

```json
{
  "id": "mp-1234",
  "formula": "MoS2",
  "text": "MoS2 is hexagonal...",
  "tensor": [11.2, 3.4, 0.1, ...],  // 9 components
  "group": "hexagonal"  // ← 추가
}
```

### 2. Config 설정

```json
{
  "model": {
    "text_backend": "huggingface",
    "text_model_name": "allenai/scibert_scivocab_uncased",
    "max_seq_length": 512,
    "clip_dim": 512
  },
  "training": {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-4
  },
  "tensor_regression": {
    "task": "elastic_2d_symmetry",
    "modality": "text",
    "tensor_mode": "voigt2d",  // 6-DOF Voigt notation
    "normalize_tensor": true,

    // Symmetry settings
    "use_symmetry_loss": true,
    "lambda_sym": 0.1,
    "lambda_sym_schedule": {
      "start": 0.05,
      "end": 0.3,
      "warmup_epochs": 50
    },
    "beta_prior": 1.0,  // Robo prior strength

    // Phase 1 settings
    "freeze_alpha_head": true,  // Freeze α-head initially
    "phase1_epochs": 30,  // Stabilization phase

    // Phase 2 settings (joint training)
    "freeze_clip": false,
    "clip_loss_weight": 0.001,

    // Loss function
    "loss_fn": "mse",  // or "mae"
    "clip_checkpoint_path": "..."
  }
}
```

### 3. Training Code 예시

```python
from src.models.elastic_regressor_2d import TextElasticRegressor2D
from src.models.elastic_loss_2d import ElasticSymmetryLoss

# Load dataset (add group field)
dataset = GraphTextDataset(
    jsonl_path="dataset_old.jsonl",
    tensor_mode="voigt2d",  # [C11, C22, C12, C66, C16, C26]
    normalize_tensor=True
)

# Build model
model = TextElasticRegressor2D(
    clip_model=clip_model,
    beta_prior=1.0,  # Prior strength
    freeze_alpha_head=True,  # Phase 1
)

# Loss function with scheduling
loss_fn = ElasticSymmetryLoss(
    lambda_sym=0.1,
    lambda_sym_schedule={
        "start": 0.05,
        "end": 0.3,
        "warmup_epochs": 50
    },
    data_loss_fn="mse"
)

# Phase 1: Stabilize tensor prediction (α-head frozen)
for epoch in range(30):
    for batch in train_loader:
        pred6, alpha, z_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            crystal_systems=batch["group"]  # Robo prior
        )

        loss, metrics = loss_fn(pred6, batch["tensor"], alpha)
        loss.backward()
        optimizer.step()

# Phase 2: Learn effective symmetry (α-head active)
model.set_freeze_alpha_head(False)  # Unfreeze

for epoch in range(30, 200):
    loss_fn.set_epoch(epoch)  # λ_sym scheduling

    for batch in train_loader:
        pred6, alpha, z_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            crystal_systems=batch["group"]
        )

        loss, metrics = loss_fn(pred6, batch["tensor"], alpha)
        loss.backward()
        optimizer.step()
```

### 4. Inference 예시

```python
model.eval()

with torch.no_grad():
    pred6, alpha, z_logits = model(
        input_ids,
        attention_mask,
        crystal_systems=["hexagonal"]
    )

# pred6: (B, 6) = [C11, C22, C12, C66, C16, C26]
# alpha: (B, 4) = [p_tri, p_rect, p_tetra, p_hexa]

# Reconstruct full 3x3 Voigt tensor
from src.models.elastic_projection_2d import reconstruct_voigt_3x3
tensor_3x3 = reconstruct_voigt_3x3(pred6)  # (B, 3, 3)
```

## Crystal System Mapping

### Robo → Prior 변환

```python
SYSTEM_TO_PRIOR = {
    "triclinic": [0.60, 0.25, 0.10, 0.05],    # [tri, rect, tetra, hexa]
    "monoclinic": [0.60, 0.25, 0.10, 0.05],
    "orthorhombic": [0.15, 0.60, 0.20, 0.05],  # Prefer rectangular
    "tetragonal": [0.10, 0.15, 0.60, 0.15],    # Prefer tetragonal
    "hexagonal": [0.05, 0.10, 0.20, 0.65],     # Prefer hexagonal
}
```

### Symmetry Constraints

| System | Constraints |
|--------|-------------|
| Triclinic | None (6-DOF) |
| Rectangular | C16 = C26 = 0 (4-DOF) |
| Tetragonal | C11 = C22, C16 = C26 = 0 (3-DOF) |
| Hexagonal | Above + C66 = (C11-C12)/2 (2-DOF) |

## Loss 구성

```
L_total = L_data + λ_sym · L_sym

L_data = ||pred6 - target6||²  (MSE or MAE)
L_sym = ||pred6 - Π(pred6)||²  (symmetry regularization)

where Π(pred6) = Σ_k α_k · Π_Gk(pred6)
```

### λ_sym Scheduling (권장)

```python
# Epoch 0-50: 0.05 → 0.3 (linear warmup)
# Epoch 50+: 0.3 (fixed)

lambda_sym_schedule = {
    "start": 0.05,
    "end": 0.3,
    "warmup_epochs": 50
}
```

## 파일 구조

```
src/models/
├── elastic_projection_2d.py    # Symmetry projections & priors
├── elastic_head_2d.py           # Dual-head architecture
├── elastic_loss_2d.py           # Loss functions
└── elastic_regressor_2d.py      # Complete model wrapper
```

## Phase 1 vs Phase 2

### Phase 1 (Stabilization, 30 epochs)
- **목표**: 안정적인 tensor 회귀 학습
- `freeze_alpha_head=True`
- `beta_prior` 크게 (1.0~2.0)
- α는 prior에 거의 고정

### Phase 2 (Symmetry Learning, 170 epochs)
- **목표**: 유효 대칭 학습
- `freeze_alpha_head=False`
- λ_sym 증가 (0.05 → 0.3)
- 모델이 prior를 correction 가능

## Sample Weighting (옵션)

C2DB 라벨이 있는 샘플에 더 높은 가중치:

```python
sample_weights = torch.where(
    has_c2db_label,
    torch.ones_like(pred6[:, 0]),  # weight = 1.0
    torch.ones_like(pred6[:, 0]) * 0.5  # weight = 0.5
)

loss, metrics = loss_fn(
    pred6, target6, alpha,
    sample_weights=sample_weights
)
```

## Troubleshooting

### α가 한 쪽으로만 몰림
- β_prior 줄이기 (1.0 → 0.5)
- Phase 1 길게 (30 → 50 epochs)
- λ_sym 천천히 증가

### 학습이 불안정
- Phase 1 먼저 충분히 수렴
- λ_sym scheduling 사용
- Learning rate 줄이기

### Prior가 너무 강함
- β_prior 줄이기
- Phase 2에서 β도 scheduling:
  ```python
  model.set_beta_prior(beta * 0.9 ** epoch)
  ```

## 참고

- Voigt notation: [C11, C22, C12, C66, C16, C26]
- Prior는 "힌트"이지 "라벨" 아님
- 모델이 prior를 override 가능해야 함
