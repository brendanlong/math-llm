# Math LLM Design Document

## Overview
A minimal transformer model to learn basic arithmetic, starting with single-digit addition. This serves as a proof-of-concept for training language models on mathematical reasoning tasks.

## Architecture Decisions

### Model Architecture
- **Type**: Decoder-only transformer (GPT-style)
- **Size**: Small model (~1M-10M parameters) to fit RTX3060 constraints
- **Context Length**: 32 tokens (sufficient for expressions like "123+456=579<end>")
- **Layers**: 4-8 transformer blocks
- **Hidden Size**: 256-512 dimensions
- **Attention Heads**: 4-8 heads

### Tokenization
- **Vocabulary**: Character-level tokenization with 12 tokens:
  - Digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - Operators: `+`, `=`
  - Special: `<end>` (sequence terminator)
- **Rationale**: Simple, interpretable, and sufficient for arithmetic tasks
- **Padding**: No padding token needed (use fixed-length sequences)

### Data Format
- **Training Examples**: `"{operand1}+{operand2}={result}<end>"`
- **Example**: `"12+34=46<end>"`
- **Progression**:
  1. Single-digit: `"3+5=8<end>"`
  2. Multi-digit: `"12+34=46<end>"`
  3. Larger numbers: `"123+456=579<end>"`

### Training Strategy
- **Framework**: HuggingFace Transformers (most stable/standard)
- **Optimizer**: AdamW with cosine learning rate schedule
- **Loss**: Cross-entropy on next-token prediction
- **Batch Size**: 32-64 (memory permitting on RTX3060)
- **Checkpointing**: Save every 1000 steps for quick restart
- **Monitoring**: Weights & Biases for metrics visualization

### Data Generation
- **Size**: 100k examples for single-digit, scale up as needed
- **Distribution**: Uniform sampling of operands within range
- **Storage**: Save to disk as JSON/CSV for reproducibility
- **Splits**: 80% train, 10% validation, 10% test

### Evaluation Metrics
- **Exact Match**: Percentage of completely correct answers
- **Token Accuracy**: Per-token accuracy during generation
- **Progression Tracking**: Accuracy by operand size/complexity

## Hardware Constraints
- **GPU**: RTX3060 (12GB VRAM)
- **Model Size Limit**: ~50M parameters maximum
- **Batch Size**: Tuned to maximize GPU utilization without OOM

## Future Extensions
- Support for subtraction, multiplication, division
- Multi-step arithmetic problems
- Larger number ranges
- Mixed operation expressions
- Chain-of-thought reasoning for complex problems