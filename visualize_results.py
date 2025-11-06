"""Quick visualization of training results."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load training results
with open('results/high_regularization_v1.json', 'r') as f:
    data = json.load(f)

epochs = [e['epoch'] for e in data['epochs']]
train_acc = [e['class_acc'] * 100 for e in data['epochs']]
val_acc = [e['avg_val_acc'] * 100 for e in data['epochs']]
domain_acc = [e['domain_acc'] * 100 for e in data['epochs']]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Classification Accuracy
ax1.plot(epochs, train_acc, label='Train Accuracy', linewidth=2, color='#2563eb')
ax1.plot(epochs, val_acc, label='Val Accuracy', linewidth=2, color='#dc2626')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

# Mark best validation
best_idx = np.argmax(val_acc)
ax1.scatter(epochs[best_idx], val_acc[best_idx], s=200, c='gold',
           marker='*', zorder=5, edgecolors='black', linewidth=2,
           label=f'Best Val: {val_acc[best_idx]:.2f}% (Epoch {epochs[best_idx]})')

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('DANN High Regularization - Classification Performance', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Add overfitting zone annotation
train_val_gap = train_acc[-1] - val_acc[-1]
ax1.fill_between(epochs, val_acc, train_acc, alpha=0.2, color='orange',
                 label=f'Train-Val Gap: {train_val_gap:.1f}%')
ax1.text(epochs[-1] - 10, (train_acc[-1] + val_acc[-1]) / 2,
         f'Gap: {train_val_gap:.1f}%', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

# Plot 2: Domain Accuracy
ax2.plot(epochs, domain_acc, label='Domain Accuracy', linewidth=2, color='#059669')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label='Target (Domain Invariance)')
ax2.fill_between(epochs, 48, 52, alpha=0.2, color='green',
                 label='Ideal Range (48-52%)')

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Domain Classification Accuracy (%)', fontsize=12)
ax2.set_title('Domain Discriminator Performance', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([45, 55])

# Add interpretation text
final_domain = domain_acc[-1]
if 48 <= final_domain <= 52:
    status = "✓ GOOD"
    color = 'green'
else:
    status = "✗ NEEDS WORK"
    color = 'red'

ax2.text(epochs[-1] - 10, final_domain,
         f'{status}\n{final_domain:.1f}%',
         fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

plt.tight_layout()
plt.savefig('results/training_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/training_analysis.png")

# Print summary statistics
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Total Epochs:           {len(epochs)}")
print(f"Best Val Accuracy:      {max(val_acc):.2f}% (Epoch {np.argmax(val_acc)})")
print(f"Final Train Accuracy:   {train_acc[-1]:.2f}%")
print(f"Final Val Accuracy:     {val_acc[-1]:.2f}%")
print(f"Train-Val Gap:          {train_acc[-1] - val_acc[-1]:.2f}%")
print(f"Final Domain Accuracy:  {domain_acc[-1]:.2f}%")
print(f"\nOverfitting Status:     {'⚠️  WARNING' if train_acc[-1] - val_acc[-1] > 15 else '✓ OK'}")
print(f"Domain Invariance:      {'✓ GOOD' if 48 <= domain_acc[-1] <= 52 else '✗ NEEDS WORK'}")
print("="*80)
