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

## Summary Results (2-Digit Addition)

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

## 5-Digit Addition: PoPE vs RoPE

To test how positional encodings scale to harder problems, we trained models on 5-digit addition (50,000 examples, 3 epochs).

| Pos Encoding | Optimizer | Test Acc | Test Loss | Hidden Kurtosis | FFN Kurtosis | Attn Entropy |
|--------------|-----------|----------:|----------:|----------------:|-------------:|-------------:|
| rope | adamw | **99.99%** | 0.0006 | 0.71 | 2.40 | 2.12 |
| rope | adopt | 99.96% | 0.0019 | 4.18 | 3.98 | 1.10 |
| pope | muon | 99.92% | 0.0032 | -0.07 | 0.73 | 2.53 |
| pope | adopt | 99.77% | 0.0077 | 1.53 | 3.55 | 2.04 |
| rope | muon | 97.30% | 0.0732 | -0.10 | 0.65 | 2.47 |
| pope | adamw | 94.38% | 0.1502 | 0.72 | 3.36 | 2.37 |

### 5-Digit Summary

| Metric | RoPE | PoPE |
|--------|-----:|-----:|
| **Average Accuracy** | **99.08%** | 98.03% |
| **Best Accuracy** | **99.99%** | 99.92% |
| Best Optimizer | AdamW | Muon |

**Key finding: RoPE outperforms PoPE on harder problems.** While PoPE achieved the best result on 2-digit addition, RoPE scales better to 5-digit addition, achieving near-perfect accuracy (99.99%) with AdamW.

### Optimizer Performance on 5-Digit Addition

| Optimizer | Avg Accuracy |
|-----------|-------------:|
| ADOPT | **99.87%** |
| Muon | 98.61% |
| AdamW | 97.19% |

Interestingly, **ADOPT performs best on the harder task**, despite showing more variability on the easier 2-digit task. This suggests ADOPT may be better suited for more challenging optimization landscapes.

## Key Findings

### 1. Optimizer Comparison (2-Digit)

With ADOPT correctly configured (`weight_decouple=True`), all three optimizers achieve comparable accuracy:

| Optimizer | Avg Hidden Kurtosis | Avg FFN Kurtosis | Avg Test Accuracy |
|-----------|--------------------:|-----------------:|------------------:|
| AdamW | 0.03 | 1.76 | 98.46% |
| ADOPT | 0.89 | 3.48 | 96.76% |
| Muon | 0.33 | 0.51 | 98.24% |

**Muon produces the lowest kurtosis** in both hidden states and FFN layers, suggesting the most uniform activation distributions.

### 2. Muon Produces the Lowest FFN Kurtosis

Muon consistently produces the **lowest FFN kurtosis** (0.44-0.73) across all configurations, while maintaining excellent accuracy. Lower kurtosis suggests more uniform activation distributions without extreme outliers.

### 3. Positional Encoding Effects Depend on Task Difficulty

| Task | Best Pos Encoding | Best Accuracy |
|------|-------------------|-------------:|
| 2-digit addition | PoPE | 99.42% |
| 5-digit addition | RoPE | 99.99% |

**RoPE scales better to harder problems**, while PoPE performs slightly better on simpler tasks.

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

When using weight decay with ADOPT, you **must** set `weight_decouple=True` to get AdamW-compatible behavior.

## Conclusions

1. **All three optimizers work well when properly configured**, with test accuracies in the 93-99%+ range
2. **Muon produces the healthiest activation distributions** with consistently low kurtosis values
3. **ADOPT requires `weight_decouple=True`** when using weight decay; without it, training diverges
4. **RoPE scales better to harder problems** (5-digit: 99.99%) while PoPE excels on simpler tasks (2-digit: 99.42%)
5. **ADOPT performs best on harder tasks** (5-digit avg: 99.87%) despite more variability on easier tasks
6. **Optimizer-positional encoding interactions matter**: The best combination depends on task difficulty

## Methodology Notes

- Kurtosis measured using Fisher's definition (normal distribution = 0)
- Statistics computed over 10 batches from validation set after training
- Hidden kurtosis averaged across all 4 layers
- FFN kurtosis measured at intermediate layer (after first linear + activation)
- ADOPT configured with `weight_decouple=True` for proper weight decay handling
- 2-digit study: 20,000 examples, 2 epochs
- 5-digit study: 50,000 examples, 3 epochs
