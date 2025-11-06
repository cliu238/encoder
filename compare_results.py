"""Compare training results between different configurations."""

import json
import sys


def compare_experiments(old_path, new_path):
    """Compare two experiment results."""

    with open(old_path, 'r') as f:
        old_data = json.load(f)

    with open(new_path, 'r') as f:
        new_data = json.load(f)

    print("="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)

    # Configuration changes
    print("\n📋 CONFIGURATION CHANGES")
    print("-"*80)
    old_config = old_data['config']
    new_config = new_data['config']

    key_params = [
        'encoder_hidden', 'encoder_output', 'embedding_dim',
        'dropout', 'weight_decay', 'lambda_adv', 'num_epochs'
    ]

    for param in key_params:
        old_val = old_config.get(param)
        new_val = new_config.get(param)
        if old_val != new_val:
            print(f"{param:20s}: {old_val:>8} → {new_val:>8}", end="")
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                change = ((new_val - old_val) / old_val) * 100
                print(f" ({change:+.0f}%)")
            else:
                print()

    # Training metrics comparison
    print("\n📊 TRAINING METRICS")
    print("-"*80)

    old_epochs = old_data['epochs']
    new_epochs = new_data['epochs']

    # Best validation accuracy
    old_best_val = max(e['avg_val_acc'] for e in old_epochs)
    new_best_val = max(e['avg_val_acc'] for e in new_epochs)
    old_best_epoch = [e['avg_val_acc'] for e in old_epochs].index(old_best_val)
    new_best_epoch = [e['avg_val_acc'] for e in new_epochs].index(new_best_val)

    print(f"Best validation accuracy:")
    print(f"  Old: {old_best_val:.2%} (epoch {old_best_epoch+1})")
    print(f"  New: {new_best_val:.2%} (epoch {new_best_epoch+1})")
    print(f"  Change: {(new_best_val - old_best_val)*100:+.2f} percentage points")

    # Final training metrics
    old_final = old_epochs[-1]
    new_final = new_epochs[-1]

    print(f"\nFinal train class accuracy:")
    print(f"  Old: {old_final['class_acc']:.2%}")
    print(f"  New: {new_final['class_acc']:.2%}")
    print(f"  Change: {(new_final['class_acc'] - old_final['class_acc'])*100:+.2f} pp")

    print(f"\nFinal domain accuracy (should be ~50%):")
    print(f"  Old: {old_final['domain_acc']:.2%}")
    print(f"  New: {new_final['domain_acc']:.2%}")
    print(f"  Distance from 50%: Old={abs(old_final['domain_acc'] - 0.5):.2%}, New={abs(new_final['domain_acc'] - 0.5):.2%}")

    # Train-val gap (overfitting indicator)
    old_gap = old_final['class_acc'] - old_final['avg_val_acc']
    new_gap = new_final['class_acc'] - new_final['avg_val_acc']

    print(f"\nTrain-val gap (overfitting indicator):")
    print(f"  Old: {old_gap:.2%}")
    print(f"  New: {new_gap:.2%}")
    print(f"  Improvement: {(old_gap - new_gap)*100:+.2f} pp {'✓ Better' if new_gap < old_gap else '✗ Worse'}")

    # Model size
    print("\n📦 MODEL SIZE")
    print("-"*80)
    old_params = (old_config['encoder_hidden'] * old_config['embedding_dim'] +
                  old_config['encoder_output'] * old_config['embedding_dim'])
    new_params = (new_config['encoder_hidden'] * new_config['embedding_dim'] +
                  new_config['encoder_output'] * new_config['embedding_dim'])

    print(f"Approximate parameter count:")
    print(f"  Old: ~{old_params:,}")
    print(f"  New: ~{new_params:,}")
    print(f"  Reduction: {((old_params - new_params) / old_params) * 100:.1f}%")

    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("-"*80)

    improvements = []
    concerns = []

    if new_gap < old_gap:
        improvements.append(f"✓ Reduced overfitting (train-val gap: {old_gap:.1%} → {new_gap:.1%})")
    else:
        concerns.append(f"✗ Increased overfitting (train-val gap: {old_gap:.1%} → {new_gap:.1%})")

    if abs(new_final['domain_acc'] - 0.5) < abs(old_final['domain_acc'] - 0.5):
        improvements.append(f"✓ Better domain confusion (closer to 50%)")
    else:
        concerns.append(f"✗ Domain accuracy still far from 50%")

    if new_best_val > old_best_val:
        improvements.append(f"✓ Higher validation accuracy ({old_best_val:.1%} → {new_best_val:.1%})")
    elif new_best_val < old_best_val:
        concerns.append(f"✗ Lower validation accuracy ({old_best_val:.1%} → {new_best_val:.1%})")

    for item in improvements:
        print(item)

    for item in concerns:
        print(item)

    if not concerns:
        print("\n🎉 All metrics improved!")
    elif len(improvements) > len(concerns):
        print("\n👍 Net positive - more improvements than concerns")
    else:
        print("\n⚠️  Need further tuning")

    print("\n" + "="*80)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <old_log.json> <new_log.json>")
        sys.exit(1)

    compare_experiments(sys.argv[1], sys.argv[2])
