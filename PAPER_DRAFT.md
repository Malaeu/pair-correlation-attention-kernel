# Neural Spectroscopy of Riemann Zeros: Learning GUE Correlations from Data

## Abstract

A small transformer trained on 2M unfolded zeta zeros learns an attention kernel μ(d) ~ d·exp(-γ√d) that captures short-range GUE-like correlations (ACF MSE=0.005), providing empirical evidence for spectral rigidity in the Riemann zero spectrum.

## Key Results

### 1. Model Architecture
- SpacingGPT: 4 layers, 4 heads, 128 embedding dim
- 0.85M parameters
- Trained on unfolded spacings (mean=1)
- 256 bins for discretization

### 2. Kernel Extraction (PySR)
Best symbolic fit (R² = 0.9927):
```
μ(d) = (0.127·d + 0.062) × exp(-1.16·√d) + 0.0017
```

Physical interpretation:
- Linear term (d + 0.48): **Level Repulsion** - nearest neighbors are constrained
- Stretched exponential exp(-γ√d): **Spectral Rigidity** - long-range correlations

### 3. Quantitative Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PySR R² | 0.9927 | Excellent kernel fit |
| ACF MSE | 0.005 | Short-range correlations learned |
| KL divergence | 0.12 | Spacing distribution matches |
| Bin accuracy | 1.6% | 4x better than random |
| Top-5 accuracy | 9.4% | 5x better than random |
| MAE (γ) | 0.136 | Moderate prediction accuracy |

### 4. What the Model Learns vs. Doesn't Learn

**Learned:**
- Short-range correlations (lag 1-50)
- Level repulsion structure
- Mean spacing normalization
- GUE-like spacing distribution

**Not fully learned:**
- Full variance (0.79x underestimate)
- Long-range SFF structure
- Exact prediction (MRE = 35%)

### 5. Q3 Spectral Gap (Independent Verification)
```
min P_A(θ) = 4.028 > c* = 1.1
Spectral gap = +2.93
```

## Conclusions

1. Neural networks can rediscover RMT structure from raw zero data
2. The attention kernel approximates sine-kernel through stretched exponential
3. Short-range GUE correlations are robustly learned (ACF MSE = 0.005)
4. Long-range structure remains challenging for finite-context models

## Figures

- `pysr_kernel.png`: PySR symbolic regression result
- `sff_honest_comparison.png`: SFF comparison
- `kernel_unfolded.png`: Attention kernel in unfolded coordinates
- `Q3_Spectral_Gap.png`: Q3 symbol verification
