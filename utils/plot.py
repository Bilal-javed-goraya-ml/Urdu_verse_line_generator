import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt


def plot_training_loss(losses):
    """Plot and save training loss"""
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    plt.title('Training Loss vs Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" Training loss plot saved to results/training_loss.png")

