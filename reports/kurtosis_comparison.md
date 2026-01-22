# Activation Kurtosis Comparison Study

This report compares how different model configurations affect the kurtosis of activations in a small transformer model trained on arithmetic tasks.

## Experimental Setup

- **Model Size**: ~2.4M parameters (4 layers, 256 hidden dim, 4 heads)
- **Training Data**: 20,000 arithmetic examples (2-digit addition with chain-of-thought)
- **Training**: 2 epochs per configuration
- **Configurations Tested**:
  - 4 positional encodings: learned, sinusoidal, rope, pope
  - 3 optimizers: AdamW, ADOPT, Muon

## Summary Results

| Model | Pos Encoding | Optimizer | Hidden Kurtosis | Hidden Max | FFN Kurtosis | FFN Max | Attn Entropy | Test Acc |
|-------|--------------|-----------|---------------:|----------:|-------------:|--------:|-------------:|---------:|
| learned-adamw | learned | adamw | 0.15 | 4.93 | 1.39 | 9.68 | 1.75 | 98.08% |
| learned-adopt | learned | adopt | **8.55** | 8.04 | **256.92** | 0.03 | 3.04 | 75.30% |
| learned-muon | learned | muon | 0.52 | 5.27 | 0.49 | 4.69 | 2.71 | **98.89%** |
| pope-adamw | pope | adamw | -0.25 | 4.44 | 1.52 | 11.07 | 2.42 | **99.24%** |
| pope-adopt | pope | adopt | 0.58 | 5.70 | **270.48** | 4.49 | 2.67 | 87.20% |
| pope-muon | pope | muon | -0.15 | 5.74 | 0.44 | 4.70 | 2.74 | 96.76% |
| rope-adamw | rope | adamw | -0.08 | 6.77 | 2.26 | 9.44 | 2.08 | 98.34% |
| rope-adopt | rope | adopt | 2.60 | 5.19 | **114.22** | 4.62 | 2.44 | 87.27% |
| rope-muon | rope | muon | 0.06 | 5.04 | 0.45 | 5.12 | 2.65 | 98.50% |
| sinusoidal-adamw | sinusoidal | adamw | 0.28 | 5.97 | 1.86 | 13.03 | 1.59 | 98.16% |
| sinusoidal-adopt | sinusoidal | adopt | 2.73 | 7.67 | **170.97** | 3.24 | 2.90 | 88.24% |
| sinusoidal-muon | sinusoidal | muon | 0.88 | 6.72 | 0.65 | 4.53 | 2.52 | 98.82% |

## Key Findings

### 1. ADOPT Creates Extremely High FFN Kurtosis

The most striking finding is that **ADOPT optimizer produces dramatically elevated FFN kurtosis** across all positional encoding types:

| Optimizer | Avg FFN Kurtosis | Avg Test Accuracy |
|-----------|----------------:|------------------:|
| AdamW | 1.75 | 98.46% |
| Muon | 0.51 | 98.24% |
| ADOPT | **203.15** | 84.50% |

ADOPT's FFN kurtosis is **~120x higher** than AdamW and **~400x higher** than Muon. This correlates with significantly worse test accuracy (~84.5% vs ~98% for other optimizers).

The extremely high kurtosis indicates ADOPT produces activations with very heavy tails (outliers) in the FFN intermediate layers. This may indicate training instability or a fundamental mismatch between ADOPT's update rules and transformer architectures.

### 2. Muon Produces the Lowest FFN Kurtosis

Muon consistently produces the **lowest FFN kurtosis** (0.44-0.65) across all positional encodings, while maintaining excellent accuracy (96.8-98.9%). Lower kurtosis suggests more uniform activation distributions without extreme outliers.

### 3. Positional Encoding Effects on Hidden Kurtosis

| Pos Encoding | Avg Hidden Kurtosis (AdamW+Muon) | Best Accuracy |
|--------------|--------------------------------:|-------------:|
| PoPE | -0.20 | 99.24% |
| RoPE | -0.01 | 98.50% |
| Learned | 0.34 | 98.89% |
| Sinusoidal | 0.58 | 98.82% |

**PoPE and RoPE produce slightly negative hidden kurtosis** (platykurtic/lighter tails), while learned and sinusoidal produce positive kurtosis (heavier tails). PoPE achieves the best accuracy (99.24% with AdamW).

### 4. Attention Entropy Patterns

AdamW tends to produce **lower attention entropy** (1.59-2.42) compared to Muon (2.52-2.74), indicating sharper attention patterns. This may indicate AdamW learns more specialized attention heads, while Muon maintains more distributed attention.

## Conclusions

1. **Optimizer choice has the largest impact on activation kurtosis**, particularly in FFN layers
2. **ADOPT should be avoided** for transformer training due to extreme kurtosis and poor convergence
3. **Muon produces the healthiest activation distributions** with low kurtosis and high accuracy
4. **PoPE positional encoding** shows slight benefits in reducing hidden state kurtosis while achieving the best accuracy
5. The correlation between high FFN kurtosis and poor accuracy suggests activation distribution monitoring could be useful for detecting training problems early

## Methodology Notes

- Kurtosis measured using Fisher's definition (normal distribution = 0)
- Statistics computed over 10 batches from validation set after training
- Hidden kurtosis averaged across all 4 layers
- FFN kurtosis measured at intermediate layer (after first linear + activation)
