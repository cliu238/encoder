"""
Monitor training progress and detect overfitting.

This script analyzes training logs in real-time and alerts you to overfitting signals:
1. Train-val accuracy gap
2. Domain accuracy divergence between train/test
3. Loss trends

Usage:
    python monitor_training.py results/20251105_161608.json
    python monitor_training.py --watch results/latest.json  # Real-time monitoring
"""

import argparse
import json
import sys
from pathlib import Path
import time


def analyze_overfitting(log_data):
    """
    Analyze training logs for overfitting signals.

    Args:
        log_data: Parsed JSON training log

    Returns:
        Dictionary with overfitting analysis
    """
    epochs = log_data.get('epochs', [])
    final_results = log_data.get('final_results', {})

    if not epochs:
        return {"error": "No epoch data found"}

    # Extract key metrics
    train_class_acc = [e['class_acc'] for e in epochs]
    val_acc = [e['avg_val_acc'] for e in epochs]
    train_domain_acc = [e['domain_acc'] for e in epochs]

    # Analysis
    analysis = {
        'total_epochs': len(epochs),
        'best_val_acc': max(val_acc),
        'best_val_epoch': val_acc.index(max(val_acc)) + 1,
        'final_train_class_acc': train_class_acc[-1],
        'final_val_acc': val_acc[-1],
        'final_train_domain_acc': train_domain_acc[-1],
    }

    # Compute gaps
    analysis['train_val_gap'] = analysis['final_train_class_acc'] - analysis['final_val_acc']

    # Test results (if available)
    if final_results:
        embedding_quality = final_results.get('embedding_quality', {})
        test_domain_acc = embedding_quality.get('domain_classification_acc', None)

        if test_domain_acc is not None:
            analysis['test_domain_acc'] = test_domain_acc
            analysis['domain_acc_jump'] = test_domain_acc - analysis['final_train_domain_acc']

    # Overfitting signals
    signals = []

    # Signal 1: Large train-val gap
    if analysis['train_val_gap'] > 0.15:
        signals.append({
            'type': 'HIGH_TRAIN_VAL_GAP',
            'severity': 'WARNING',
            'message': f"Train-val accuracy gap is {analysis['train_val_gap']:.1%} (>15%)",
            'recommendation': "Increase dropout or weight_decay"
        })

    # Signal 2: Domain accuracy jump train→test
    if 'domain_acc_jump' in analysis and analysis['domain_acc_jump'] > 0.2:
        signals.append({
            'type': 'DOMAIN_ACCURACY_JUMP',
            'severity': 'CRITICAL',
            'message': f"Domain accuracy jumped {analysis['domain_acc_jump']:.1%} from train ({analysis['final_train_domain_acc']:.1%}) to test ({analysis['test_domain_acc']:.1%})",
            'recommendation': "Model overfitted! Reduce model capacity or increase lambda_adv"
        })

    # Signal 3: Test domain accuracy too high (domain adaptation failed)
    if 'test_domain_acc' in analysis and analysis['test_domain_acc'] > 0.7:
        signals.append({
            'type': 'DOMAIN_ADAPTATION_FAILED',
            'severity': 'CRITICAL',
            'message': f"Test domain accuracy is {analysis['test_domain_acc']:.1%} (should be ~50%)",
            'recommendation': "Increase lambda_adv (try 3.0 or 5.0)"
        })

    # Signal 4: Val accuracy peaked early
    if analysis['best_val_epoch'] < len(epochs) * 0.3:
        signals.append({
            'type': 'EARLY_PEAK',
            'severity': 'WARNING',
            'message': f"Validation accuracy peaked at epoch {analysis['best_val_epoch']} (early in training)",
            'recommendation': "Model may be underfitting or learning rate too high"
        })

    # Signal 5: Train domain accuracy too high (adversarial training weak)
    if analysis['final_train_domain_acc'] > 0.65:
        signals.append({
            'type': 'WEAK_ADVERSARIAL_TRAINING',
            'severity': 'WARNING',
            'message': f"Train domain accuracy is {analysis['final_train_domain_acc']:.1%} (should be ~50%)",
            'recommendation': "Increase lambda_adv or change alpha_schedule to 'const'"
        })

    analysis['overfitting_signals'] = signals

    return analysis


def print_analysis(analysis):
    """Pretty print overfitting analysis."""
    print("\n" + "="*80)
    print("TRAINING OVERFITTING ANALYSIS")
    print("="*80)

    print("\n📊 SUMMARY")
    print("-"*80)
    print(f"Total epochs:           {analysis['total_epochs']}")
    print(f"Best val accuracy:      {analysis['best_val_acc']:.2%} (epoch {analysis['best_val_epoch']})")
    print(f"Final train class acc:  {analysis['final_train_class_acc']:.2%}")
    print(f"Final val acc:          {analysis['final_val_acc']:.2%}")
    print(f"Train-val gap:          {analysis['train_val_gap']:.2%}")

    if 'test_domain_acc' in analysis:
        print(f"\nFinal train domain acc: {analysis['final_train_domain_acc']:.2%}")
        print(f"Test domain acc:        {analysis['test_domain_acc']:.2%}")
        print(f"Domain acc jump:        {analysis['domain_acc_jump']:+.2%}")

    # Print overfitting signals
    signals = analysis.get('overfitting_signals', [])

    if not signals:
        print("\n✅ NO OVERFITTING SIGNALS DETECTED")
        print("Model generalization looks good!")
    else:
        print(f"\n⚠️  DETECTED {len(signals)} OVERFITTING SIGNAL(S)")
        print("-"*80)

        for i, signal in enumerate(signals, 1):
            severity_icon = "🔴" if signal['severity'] == 'CRITICAL' else "🟡"
            print(f"\n{severity_icon} Signal {i}: {signal['type']}")
            print(f"   Severity: {signal['severity']}")
            print(f"   {signal['message']}")
            print(f"   💡 Recommendation: {signal['recommendation']}")

    print("\n" + "="*80)


def watch_log(log_path, interval=5):
    """Watch log file and update analysis in real-time."""
    print(f"👀 Watching {log_path} for updates (Ctrl+C to stop)...")
    print(f"Update interval: {interval} seconds\n")

    last_epoch_count = 0

    try:
        while True:
            if not log_path.exists():
                print(f"Waiting for {log_path} to be created...")
                time.sleep(interval)
                continue

            with open(log_path, 'r') as f:
                data = json.load(f)

            epochs = data.get('epochs', [])

            if len(epochs) > last_epoch_count:
                # New epoch data
                last_epoch_count = len(epochs)

                # Print latest epoch
                latest = epochs[-1]
                print(f"Epoch {latest['epoch']+1:3d} | "
                      f"ClassAcc: {latest['class_acc']:.3f} | "
                      f"ValAcc: {latest['avg_val_acc']:.3f} | "
                      f"DomainAcc: {latest['domain_acc']:.3f} | "
                      f"Alpha: {latest['alpha']:.3f}")

            # Check if training is complete
            if 'final_results' in data and 'end_time' in data:
                print("\n✓ Training complete! Running final analysis...\n")
                analysis = analyze_overfitting(data)
                print_analysis(analysis)
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped watching. Running analysis on current data...\n")
        with open(log_path, 'r') as f:
            data = json.load(f)
        analysis = analyze_overfitting(data)
        print_analysis(analysis)


def main():
    parser = argparse.ArgumentParser(description='Monitor training for overfitting')
    parser.add_argument('log_file', type=str, help='Path to training log JSON file')
    parser.add_argument('--watch', action='store_true', help='Watch file in real-time')
    parser.add_argument('--interval', type=int, default=5, help='Update interval for watching (seconds)')

    args = parser.parse_args()

    log_path = Path(args.log_file)

    if args.watch:
        watch_log(log_path, args.interval)
    else:
        if not log_path.exists():
            print(f"Error: {log_path} does not exist")
            sys.exit(1)

        with open(log_path, 'r') as f:
            data = json.load(f)

        analysis = analyze_overfitting(data)
        print_analysis(analysis)


if __name__ == '__main__':
    main()
