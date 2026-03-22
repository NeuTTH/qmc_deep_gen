"""
Quick check of model training status.

Usage:
    python quick_training_check.py --model_path="path/to/checkpoint.tar"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import fire


def check_training(model_path):
    """Check training progress from checkpoint."""

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    losses = np.array(checkpoint['run_info'])

    print("\n" + "="*60)
    print("TRAINING STATUS")
    print("="*60)
    print(f"Total epochs: {len(losses)}")
    print(f"Initial loss (evidence): {-losses[0]:.2f}")
    print(f"Final loss (evidence): {-losses[-1]:.2f}")
    print(f"Total improvement: {losses[-1] - losses[0]:.2f}")

    # Check if still improving
    if len(losses) > 100:
        last_100 = losses[-100:]
        trend = np.polyfit(np.arange(len(last_100)), last_100, 1)[0]
        print(f"\nRecent trend (last 100 epochs): {trend:.4f} per epoch")

        if trend > 0.01:
            print("✓ Still improving - consider training longer!")
        elif trend > -0.001:
            print("~ Plateaued - likely converged")
        else:
            print("⚠ Degrading - check for issues")

    # Plot training curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Full curve
    ax = axes[0]
    ax.plot(-losses, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Log Evidence', fontsize=12)
    ax.set_title('Full Training Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Last 20% (to see recent behavior)
    ax = axes[1]
    last_20pct = int(len(losses) * 0.2)
    if last_20pct > 0:
        ax.plot(-losses[-last_20pct:], linewidth=1.5, color='orange')
        ax.set_xlabel('Epoch (last 20%)', fontsize=12)
        ax.set_ylabel('Log Evidence', fontsize=12)
        ax.set_title('Recent Training', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print(f"\n✓ Saved: training_curve.png")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if len(losses) < 100:
        print("⚠ Very few epochs - train much longer!")
        print(f"  Current: {len(losses)} epochs")
        print(f"  Suggested: 200-500 epochs")
    elif len(losses) < 200:
        print("⚠ Model may need more training")
        print(f"  Current: {len(losses)} epochs")
        print(f"  Try: 300-500 epochs")
    else:
        if trend > 0.005:
            print("✓ Keep training - still improving")
        else:
            print("✓ Training looks complete")

    print("="*60)


if __name__ == '__main__':
    fire.Fire(check_training)
