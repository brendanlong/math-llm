# Activation Kurtosis Comparison Study

This report compares how different model configurations affect the kurtosis of activations in a small transformer model trained on arithmetic tasks.

## Experimental Setup

- **Model Size**: ~2.4M parameters (4 layers, 256 hidden dim, 4 heads)
- **Training Data**: 20,000 arithmetic examples (2-digit addition with chain-of-thought)
- **Training**: 2 epochs per configuration
- **Configurations Tested**:
  - 4 positional encodings: learned, sinusoidal, rope, pope
  - 3 optimizers: AdamW, ADOPT, Muon

**Note on ADOPT configuration**: ADOPT requires `weight_decouple=True` when using weight decay (to match AdamW behavior). The results below use the correctly configured ADOPT optimizer.

## Summary Results

| Model | Pos Encoding | Optimizer | Hidden Kurtosis | Hidden Max | FFN Kurtosis | FFN Max | Attn Entropy | Test Acc |
|-------|--------------|-----------|---------------:|----------:|-------------:|--------:|-------------:|---------:|
| learned-adamw | learned | adamw | 0.15 | 4.93 | 1.39 | 9.68 | 1.75 | 98.08% |
| learned-adopt | learned | adopt | 1.49 | 11.19 | 2.47 | 11.93 | 0.89 | 98.12% |
| learned-muon | learned | muon | 0.52 | 5.27 | 0.49 | 4.69 | 2.71 | 98.89% |
| pope-adamw | pope | adamw | -0.25 | 4.44 | 1.52 | 11.07 | 2.42 | 99.24% |
| pope-adopt | pope | adopt | 0.54 | 7.19 | 3.05 | 13.29 | 2.14 | **99.42%** |
| pope-muon | pope | muon | -0.15 | 5.74 | 0.44 | 4.70 | 2.74 | 96.76% |
| rope-adamw | rope | adamw | -0.08 | 6.77 | 2.26 | 9.44 | 2.08 | 98.34% |
| rope-adopt | rope | adopt | 0.78 | 9.71 | 4.37 | 17.22 | 0.98 | 93.32% |
| rope-muon | rope | muon | 0.06 | 5.04 | 0.45 | 5.12 | 2.65 | 98.50% |
| sinusoidal-adamw | sinusoidal | adamw | 0.28 | 5.97 | 1.86 | 13.03 | 1.59 | 98.16% |
| sinusoidal-adopt | sinusoidal | adopt | 0.76 | 7.82 | 4.04 | 17.33 | 0.48 | 96.19% |
| sinusoidal-muon | sinusoidal | muon | 0.88 | 6.72 | 0.65 | 4.53 | 2.52 | 98.82% |

## Key Findings

### 1. Optimizer Comparison

With ADOPT correctly configured (`weight_decouple=True`), all three optimizers achieve comparable accuracy:

| Optimizer | Avg Hidden Kurtosis | Avg FFN Kurtosis | Avg Test Accuracy |
|-----------|--------------------:|-----------------:|------------------:|
| AdamW | 0.03 | 1.76 | 98.46% |
| ADOPT | 0.89 | 3.48 | 96.76% |
| Muon | 0.33 | 0.51 | 98.24% |

**Muon produces the lowest kurtosis** in both hidden states and FFN layers, suggesting the most uniform activation distributions.

**ADOPT produces slightly higher kurtosis** than AdamW/Muon, and shows more variability across positional encodings (93-99% accuracy range vs 97-99% for others).

### 2. Muon Produces the Lowest FFN Kurtosis

Muon consistently produces the **lowest FFN kurtosis** (0.44-0.65) across all positional encodings, while maintaining excellent accuracy (96.8-98.9%). Lower kurtosis suggests more uniform activation distributions without extreme outliers.

### 3. Positional Encoding Effects

| Pos Encoding | Avg Hidden Kurtosis | Avg FFN Kurtosis | Best Accuracy |
|--------------|--------------------:|-----------------:|-------------:|
| PoPE | 0.05 | 1.67 | **99.42%** |
| RoPE | 0.25 | 2.36 | 98.50% |
| Learned | 0.72 | 1.45 | 98.89% |
| Sinusoidal | 0.64 | 2.18 | 98.82% |

**PoPE achieves the best overall accuracy** (99.42% with ADOPT) while maintaining low hidden kurtosis. The combination of PoPE + ADOPT produced the single best result in this study.

### 4. Attention Entropy Patterns

| Optimizer | Avg Attention Entropy |
|-----------|----------------------:|
| AdamW | 1.96 |
| ADOPT | 1.12 |
| Muon | 2.66 |

**ADOPT produces much lower attention entropy** (sharper, more focused attention) compared to other optimizers. Muon produces the highest entropy (more distributed attention).

### 5. ADOPT Configuration Matters

ADOPT requires careful configuration when replacing AdamW:

| ADOPT Configuration | Test Accuracy | FFN Kurtosis |
|--------------------|-------------:|--------------:|
| `weight_decouple=False` (incorrect) | 75-88% | 114-270 |
| `weight_decouple=True` (correct) | 93-99% | 2.5-4.4 |

When using weight decay with ADOPT, you **must** set `weight_decouple=True` to get AdamW-compatible behavior. Without this, ADOPT fails to converge properly and produces extremely high kurtosis values.

## Conclusions

1. **All three optimizers work well when properly configured**, with test accuracies in the 96-99% range
2. **Muon produces the healthiest activation distributions** with consistently low kurtosis values
3. **ADOPT requires `weight_decouple=True`** when using weight decay; without it, training diverges
4. **ADOPT shows more sensitivity to positional encoding choice** (wider accuracy range) than AdamW or Muon
5. **PoPE positional encoding** achieved the best single result (99.42% with ADOPT)
6. **PoPE and RoPE are comparable** - neither consistently outperforms the other across all optimizers

## Methodology Notes

- Kurtosis measured using Fisher's definition (normal distribution = 0)
- Statistics computed over 10 batches from validation set after training
- Hidden kurtosis averaged across all 4 layers
- FFN kurtosis measured at intermediate layer (after first linear + activation)
- ADOPT configured with `weight_decouple=True` for proper weight decay handling
