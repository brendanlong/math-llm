# Math LLM Design Document

## Overview

A minimal transformer model to learn basic arithmetic, starting with single-digit addition. This serves as a proof-of-concept for training language models on mathematical reasoning tasks.

## Architecture Decisions

### Model Architecture

- **Type**: Decoder-only transformer (GPT-style)
- **Size**: Small model (~1M-10M parameters) to fit RTX3060 constraints
- **Context Length**: 128 tokens (sufficient for chain-of-thought reasoning in expressions like "658+189=<think>...<\/think>847<end>")
- **Layers**: 4-8 transformer blocks
- **Hidden Size**: 256-512 dimensions
- **Attention Heads**: 4-8 heads

### Tokenization

- **Vocabulary**: Character-level tokenization with 16 tokens:
  - Digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - Operators: `+`, `=`
  - Special: `<end>` (sequence terminator)
  - Reasoning: `<think>`, `</think>` (chain-of-thought delimiters)
  - Formatting: `\n` (newline for reasoning steps)
- **Rationale**: Simple, interpretable, and supports explicit reasoning
- **Padding**: Uses end tokens for fixed-length sequences

### Data Format

- **Training Examples**: Support both simple and reasoning formats
- **Simple Examples**: `"3+5=8<end>"` (single-digit, no reasoning needed)
- **Reasoning Examples**: `"658+189=<think>\n8+9=17\n5+8=14\n6+1=8</think>847<end>"`
- **Chain-of-Thought**: Shows step-by-step column addition for multi-digit problems
- **Progression**:
  1. Single-digit: `"3+5=8<end>"`
  2. Multi-digit with reasoning: `"12+34=<think>\n2+4=6\n1+3=4</think>46<end>"`
  3. Complex problems: `"658+189=<think>\n8+9=17\n5+8=14\n6+1=8</think>847<end>"`

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

### Chain-of-Thought Reasoning

- **Approach**: Explicit step-by-step reasoning for multi-digit addition
- **Format**: `<think>` and `</think>` delimiters around reasoning steps
- **Steps**: Shows column addition from right to left with carry operations
- **Benefits**:
  - Improved interpretability of model decisions
  - Better generalization to larger numbers
  - Explicit modeling of arithmetic procedures
- **Automatic Generation**: Only applied when problems require multi-digit addition

### Evaluation Metrics

- **Exact Match**: Percentage of completely correct answers
- **Token Accuracy**: Per-token accuracy during generation
- **Reasoning Quality**: Correctness of intermediate reasoning steps
- **Progression Tracking**: Accuracy by operand size/complexity

## Hardware Constraints

- **GPU**: RTX3060 Ti (8GB VRAM)
- **Model Size Limit**: ~50M parameters maximum
- **Batch Size**: Tuned to maximize GPU utilization without OOM

## Future Extensions

- Support for subtraction, multiplication, division
- Multi-step arithmetic problems
- Larger number ranges
- Mixed operation expressions
- Advanced reasoning patterns (e.g., borrowing for subtraction)
- Interactive reasoning exploration and debugging
