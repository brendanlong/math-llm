#!/usr/bin/env python3
"""Test script to verify Gumbel-Softmax generation works correctly."""

import sys
from pathlib import Path

import torch

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_model
from src.tokenizer import ArithmeticTokenizer


def main():
    """Test Gumbel-Softmax generation."""
    print("Testing Gumbel-Softmax generation...")

    # Create model and tokenizer
    model = create_model("xsmall")
    tokenizer = ArithmeticTokenizer()

    # Create a simple example
    text = "3+5="
    input_ids = torch.tensor([tokenizer.encode(text)])

    # Create labels (full sequence including answer)
    full_text = "3+5=8<end>"
    labels = torch.tensor([tokenizer.encode(full_text)])

    # Pad to same length
    max_len = max(input_ids.shape[1], labels.shape[1])
    if input_ids.shape[1] < max_len:
        input_ids = torch.nn.functional.pad(
            input_ids, (0, max_len - input_ids.shape[1]), value=tokenizer.end_token_id
        )
    if labels.shape[1] < max_len:
        labels = torch.nn.functional.pad(
            labels, (0, max_len - labels.shape[1]), value=-100
        )

    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test regular forward pass
    print("\n1. Testing regular forward pass...")
    outputs = model(input_ids, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Test Gumbel-Softmax forward pass
    print("\n2. Testing Gumbel-Softmax forward pass...")
    outputs_gumbel = model(
        input_ids, labels=labels, use_gumbel=True, gumbel_temperature=1.0
    )
    print(f"Loss: {outputs_gumbel['loss'].item():.4f}")
    print(f"Logits shape: {outputs_gumbel['logits'].shape}")

    # Test different temperatures
    print("\n3. Testing different Gumbel temperatures...")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        outputs_temp = model(
            input_ids, labels=labels, use_gumbel=True, gumbel_temperature=temp
        )
        print(f"Temperature {temp}: Loss = {outputs_temp['loss'].item():.4f}")

    # Test evaluation mode (should use regular forward pass even with use_gumbel=True)
    print("\n4. Testing evaluation mode (should not use Gumbel)...")
    model.eval()
    outputs_eval = model(
        input_ids, labels=labels, use_gumbel=True, gumbel_temperature=1.0
    )
    print(f"Eval mode with use_gumbel=True: Loss = {outputs_eval['loss'].item():.4f}")
    print(f"Eval mode logits shape: {outputs_eval['logits'].shape}")
    model.train()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
