# Math LLM Design Document

## Overview

A minimal transformer model to learn basic arithmetic, starting with single-digit addition. This serves as a proof-of-concept for training language models on mathematical reasoning tasks.

## Architecture Decisions

### Model Architecture

- **Type**: Decoder-only transformer (GPT-style), with Universal Transformer (weight sharing), Feedback Transformer, and Mamba-style SSM variants
- **Size**: Small model (~1M-10M parameters) to fit RTX 3060 constraints
- **Context Length**: 128 tokens (sufficient for chain-of-thought reasoning)
- **Layers**: 4-8 transformer blocks
- **Hidden Size**: 256-512 dimensions
- **Attention Heads**: 4-8 heads
- **Layer Norm**: Configurable pre-LN or post-LN placement

### Tokenization

- **Vocabulary**: Character-level tokenization with 17 tokens:
  - Digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - Operators: `+`, `=`
  - Special: `<begin>` (sequence start), `<end>` (sequence terminator), `<noop>` (no operation)
  - Reasoning: `<think>`, `</think>` (chain-of-thought reasoning)
- **Rationale**: Simple, interpretable, and supports explicit reasoning
- **Padding**: Uses end tokens for fixed-length sequences

### Data Format

- **Training Examples**: Support both simple and reasoning formats; every example starts with `<begin>`
- **Simple Examples**: `"<begin>3+5=8<end>"` (no reasoning)
- **Reasoning Examples**: `"<begin>99+21=<think>99+12=021</think>120<end>"`
- **Chain-of-Thought**: Recursive pairwise addition with digits reversed inside `<think>` (least-significant digit first, which matches the left-to-right generation order); the final answer is in normal digit order
- **Reversed Format** (optional, no CoT): `"<begin>8+21=02<end>"` — digits reversed in operands and result

### Training Strategy

- **Framework**: HuggingFace Transformers (most stable/standard)
- **Optimizer**: AdamW by default (ADOPT and Muon also available)
- **Loss**: Cross-entropy on completion-style training (only predict tokens after "=")
- **Batch Size**: 32-64 (memory permitting on RTX 3060)
- **Checkpointing**: Save every 1000 steps for quick restart (`model.safetensors`)
- **Monitoring**: Weights & Biases for metrics visualization

### Data Generation

- **Size**: 100k examples for single-digit, scale up as needed
- **Distribution**: Uniform sampling of digit counts, then uniform operands within range
- **Storage**: Save to disk as JSONL (one example per line) for reproducibility
- **Reproducibility**: Output depends only on the seed and parameters, not on worker count
- **Splits**: 80% train, 10% validation, 10% test; all copies of a duplicated expression are assigned to the same split to avoid train/test leakage

### Chain-of-Thought Reasoning

- **Approach**: Explicit step-by-step reasoning for multi-operand addition
- **Format**: `<think>`/`</think>` delimiters for chain-of-thought reasoning
- **Steps**: Recursive pairwise addition with digit-reversed numbers (least-significant first), so the model can emit digits in the order it computes them
- **Benefits**:
  - Improved interpretability of model decisions
  - Better generalization to larger numbers
  - Explicit modeling of arithmetic procedures

### Evaluation Metrics

- **Exact Match**: Percentage of completely correct answers
- **Token Accuracy**: Per-token accuracy during generation
- **Reasoning Quality**: Correctness of intermediate reasoning steps
- **Progression Tracking**: Accuracy by operand size/complexity

## Hardware Constraints

- **GPU**: RTX 3060 (8GB VRAM)
- **Model Size Limit**: ~50M parameters maximum
- **Batch Size**: Tuned to maximize GPU utilization without OOM

## Future Extensions

- Support for subtraction, multiplication, division
- Multi-step arithmetic problems
- Larger number ranges
- Mixed operation expressions
- Advanced reasoning patterns (e.g., borrowing for subtraction)
- Interactive reasoning exploration and debugging
