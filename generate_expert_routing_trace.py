#!/usr/bin/env python3
"""
Generate Expert Routing Trace for MoE Simulation

Creates a CSV file with token-to-expert routing probabilities.
Supports various distribution patterns: uniform, zipf (skewed), custom.

Usage:
    python generate_expert_routing_trace.py --batch_size 512 --num_experts 8 --skew 0.8
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def generate_zipf_distribution(num_experts, skew_factor):
    """
    Generate Zipf-like distribution (skewed towards top experts)
    
    Args:
        num_experts: Total number of experts
        skew_factor: Concentration parameter (0.5-0.95)
                     0.8 = 80% weight on top 20% experts
    
    Returns:
        Array of probabilities (sums to 1.0)
    """
    # Create Zipf distribution: p(k) ~ 1/k^alpha
    # Higher alpha = more skewed
    alpha = -np.log(1 - skew_factor) / np.log(2)  # Map skew to alpha
    
    ranks = np.arange(1, num_experts + 1)
    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()  # Normalize to sum to 1.0
    
    return probs


def generate_uniform_distribution(num_experts):
    """Uniform distribution - all experts equally likely"""
    return np.ones(num_experts) / num_experts


def generate_power_law_distribution(num_experts, concentration):
    """
    Power law distribution
    
    Args:
        concentration: How concentrated on top experts (1.0-3.0)
                      1.0 = mild concentration, 3.0 = extreme
    """
    ranks = np.arange(1, num_experts + 1)
    probs = 1.0 / (ranks ** concentration)
    probs = probs / probs.sum()
    return probs


def add_noise(probs, noise_level=0.05):
    """Add random noise to probabilities (makes it more realistic)"""
    noise = np.random.uniform(-noise_level, noise_level, len(probs))
    probs_noisy = probs + noise
    probs_noisy = np.maximum(probs_noisy, 0)  # Ensure non-negative
    probs_noisy = probs_noisy / probs_noisy.sum()  # Re-normalize
    return probs_noisy


def generate_routing_trace(
    batch_size,
    num_experts,
    num_layers=1,
    distribution='zipf',
    skew_factor=0.8,
    concentration=2.0,
    add_token_noise=True,
    noise_level=0.05,
    output_path='expert_routing_trace.csv'
):
    """
    Generate expert routing trace file
    
    Args:
        batch_size: Number of tokens per batch
        num_experts: Total number of experts
        num_layers: Number of transformer layers
        distribution: 'zipf', 'uniform', 'power_law', or 'custom'
        skew_factor: For zipf (0.5-0.95, higher = more skewed)
        concentration: For power_law (1.0-3.0, higher = more concentrated)
        add_token_noise: Add per-token randomness
        noise_level: Amount of noise (0.0-0.2)
        output_path: Where to save the CSV file
    
    Returns:
        Path to generated file
    """
    
    print(f"Generating MoE routing trace:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num experts: {num_experts}")
    print(f"  Num layers: {num_layers}")
    print(f"  Distribution: {distribution}")
    
    # Generate base distribution
    if distribution == 'zipf':
        base_probs = generate_zipf_distribution(num_experts, skew_factor)
        print(f"  Skew factor: {skew_factor}")
    elif distribution == 'uniform':
        base_probs = generate_uniform_distribution(num_experts)
    elif distribution == 'power_law':
        base_probs = generate_power_law_distribution(num_experts, concentration)
        print(f"  Concentration: {concentration}")
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    print(f"\nBase expert probabilities:")
    for i, p in enumerate(base_probs[:10]):
        print(f"  Expert {i}: {p:.4f} ({p*100:.1f}%)")
    if num_experts > 10:
        print(f"  ... ({num_experts - 10} more experts)")
    
    # Generate trace data
    rows = []
    for layer_id in range(num_layers):
        for token_id in range(batch_size):
            # Start with base distribution
            token_probs = base_probs.copy()
            
            # Add per-token variation
            if add_token_noise:
                token_probs = add_noise(token_probs, noise_level)
            
            # Create row: layer_id, token_id, expert_0, expert_1, ...
            row = [layer_id, token_id] + token_probs.tolist()
            rows.append(row)
    
    # Create DataFrame
    columns = ['layer_id', 'token_id'] + [f'expert_{i}' for i in range(num_experts)]
    df = pd.DataFrame(rows, columns=columns)
    
    # Save to CSV
    output_path = Path(output_path)
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n✓ Saved routing trace to: {output_path}")
    print(f"  Total rows: {len(df)} (layers × tokens)")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print statistics
    print(f"\nToken Assignment Statistics:")
    for layer_id in range(min(3, num_layers)):  # Show first 3 layers
        layer_data = df[df['layer_id'] == layer_id]
        expert_cols = [f'expert_{i}' for i in range(num_experts)]
        
        # Simulate top-2 selection
        top_experts = []
        for _, row in layer_data.iterrows():
            probs = row[expert_cols].values
            top_2_indices = np.argsort(probs)[-2:][::-1]
            top_experts.extend(top_2_indices)
        
        # Count assignments
        unique, counts = np.unique(top_experts, return_counts=True)
        
        print(f"\n  Layer {layer_id} (top-2 selection):")
        for expert_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
            pct = 100.0 * count / len(top_experts)
            print(f"    Expert {expert_id}: {count} assignments ({pct:.1f}%)")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate MoE expert routing trace')
    
    # Required parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of tokens per batch (default: 512)')
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts (default: 8)')
    
    # Optional parameters
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer layers (default: 1)')
    parser.add_argument('--distribution', type=str, default='zipf',
                        choices=['zipf', 'uniform', 'power_law'],
                        help='Distribution type (default: zipf)')
    parser.add_argument('--skew', type=float, default=0.8,
                        help='Skew factor for zipf distribution (0.5-0.95, default: 0.8)')
    parser.add_argument('--concentration', type=float, default=2.0,
                        help='Concentration for power_law (1.0-3.0, default: 2.0)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Per-token noise level (0.0-0.2, default: 0.05)')
    parser.add_argument('--output', type=str, default='expert_routing_trace.csv',
                        help='Output file path (default: expert_routing_trace.csv)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.skew < 0.5 or args.skew > 0.95:
        print(f"Warning: skew={args.skew} outside recommended range [0.5, 0.95]")
    if args.concentration < 1.0 or args.concentration > 3.0:
        print(f"Warning: concentration={args.concentration} outside recommended range [1.0, 3.0]")
    if args.noise < 0.0 or args.noise > 0.2:
        print(f"Warning: noise={args.noise} outside recommended range [0.0, 0.2]")
    
    # Generate trace
    trace_path = generate_routing_trace(
        batch_size=args.batch_size,
        num_experts=args.num_experts,
        num_layers=args.num_layers,
        distribution=args.distribution,
        skew_factor=args.skew,
        concentration=args.concentration,
        add_token_noise=args.noise > 0,
        noise_level=args.noise,
        output_path=args.output
    )
    
    print(f"\n✅ Done! Use this file in your config:")
    print(f'   "moe_routing_trace_path": "./{trace_path}"')


if __name__ == '__main__':
    main()

