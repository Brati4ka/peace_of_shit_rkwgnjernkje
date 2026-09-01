# Классификатор: дерево решений + временное сглаживание (NIGHT -> FOG -> SMOKE -> RAIN -> OFF). Оценка качества, матрица ошибок, эксперимент по варьированию порога ±20%.
import numpy as np
import json
from collections import deque, Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
LABELS = ['OFF', 'NIGHT', 'FOG', 'SMOKE', 'RAIN']

""" SceneAnalyzer class and related functions for classifying scenes based on extracted features.
This module defines the SceneAnalyzer class, which implements a rule-based classifier for scene classification based on extracted features. It also provides functions to classify datasets, evaluate the classifier, plot confusion matrices, and"""
class SceneAnalyzer:
    def __init__(self, thresholds, window=5, vote=3):
        self.thresholds = thresholds
        self.window = window
        self.vote = vote
        self.history = deque(maxlen=window)
        self.current_mode = "OFF"

    def classify_frame(self, features):
        b, s, lv, ed, dr = features

        if b < self.thresholds["night_brightness"] or b > 55:
            return "NIGHT"

        if (b >= self.thresholds["fog_brightness_min"]
                and s <= self.thresholds["fog_std_max"]
                and lv <= self.thresholds["fog_laplac_max"]):
            return "FOG"

        sb_lo, sb_hi = self.thresholds["smoke_brightness_range"]
        if (sb_lo <= b <= sb_hi
                and s <= self.thresholds["smoke_std_max"]
                and lv <= self.thresholds["smoke_laplac_max"]):
            return "SMOKE"

        if (ed >= self.thresholds["rain_edge_density_min"]
                and dr >= self.thresholds["rain_diagonal_ratio_min"]):
            return "RAIN"

        return "OFF"

    def update_history(self, raw_mode):
        self.history.append(raw_mode)
        if len(self.history) >= self.window:
            counter = Counter(self.history)
            mode, freq = counter.most_common(1)[0]
            if freq >= self.vote:
                self.current_mode = mode
        return self.current_mode

    def classify_frame_smoothed(self, features):
        raw = self.classify_frame(features)
        return self.update_history(raw)


def classify_dataset(df, thresholds, smoothed=False, window=5, vote=3):
    classifier = SceneAnalyzer(thresholds, window=window, vote=vote)
    predictions = []
    history = deque(maxlen=window)

    for _, row in df.iterrows():
        features = (row['brightness'], row['std'], row['laplac_var'],
                    row['edge_density'], row['diag_ratio'])

        raw = classifier.classify_frame(features)

        if smoothed:
            history.append(raw)
            if len(history) >= window:
                mode, freq = Counter(history).most_common(1)[0]
                pred = mode if freq >= vote else raw
            else:
                pred = raw
        else:
            pred = raw

        predictions.append(pred)

    return predictions


def evaluate_classifier(df, thresholds, window=5, vote=3, smoothed=False):
    y_true = df['class'].values
    y_pred = classify_dataset(df, thresholds, smoothed=smoothed,
                              window=window, vote=vote)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, (cm / row_sums) * 100, 0.0)
    cm_pct = np.round(cm_pct, 1)

    total = cm.sum()
    overall_accuracy = (cm.diagonal().sum() / total * 100) if total > 0 else 0.0

    class_accuracies = {}
    for i, label in enumerate(LABELS):
        s = cm[i].sum()
        class_accuracies[label] = {
            'accuracy_pct': float((cm[i][i] / s * 100) if s > 0 else 0.0),
            'n': int(s)
        }

    return {
        'confusion_matrix': cm,
        'confusion_matrix_pct': cm_pct,
        'overall_accuracy': float(overall_accuracy),
        'class_accuracies': class_accuracies,
        'labels': list(LABELS),
        'predictions': y_pred,
    }


def plot_confusion_matrix(eval_results, title="Confusion Matrix", out_path='confusion_matrix.png'):
    cm_pct = eval_results['confusion_matrix_pct']
    labels = eval_results['labels']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f'{cm_pct[i, j]:.1f}',
                    ha='center', va='center',
                    color='white' if cm_pct[i, j] > 50 else 'black',
                    fontsize=10)

    fig.colorbar(im, ax=ax, label='% of true class')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def experiment_threshold(df, thresholds, threshold_name, base_value,
                         range_pct=20, num_steps=5, smoothed=False):
    results = []
    lo = base_value * (1 - range_pct / 100)
    hi = base_value * (1 + range_pct / 100)
    values = np.linspace(lo, hi, num_steps)

    for value in values:
        test_thresholds = {k: (v.copy() if isinstance(v, list) else v)
                           for k, v in thresholds.items()}

        if threshold_name == 'smoke_brightness_range':
            test_thresholds[threshold_name] = [float(value),
                                               thresholds[threshold_name][1]]
        elif threshold_name == 'smoke_brightness_range_max':
            test_thresholds['smoke_brightness_range'] = [
                thresholds['smoke_brightness_range'][0], float(value)]
            threshold_name_for_print = 'smoke_brightness_range[1]'
        else:
            test_thresholds[threshold_name] = float(value)

        eval_results = evaluate_classifier(df, test_thresholds, smoothed=smoothed)
        results.append({
            'threshold_value': float(value),
            'overall_accuracy': eval_results['overall_accuracy'],
            'class_accuracies': eval_results['class_accuracies']
        })

    return results


def plot_experiment_results(results, threshold_name, out_path='experiment_results.png'):
    values = [r['threshold_value'] for r in results]
    overall = [r['overall_accuracy'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(values, overall, 'bo-', linewidth=2, markersize=8)
    mid_idx = len(results) // 2
    plt.axvline(x=results[mid_idx]['threshold_value'],
                color='red', linestyle='--', alpha=0.5, label='Original value')
    plt.xlabel(f'{threshold_name} Value')
    plt.ylabel('Overall Accuracy (%)')
    plt.title(f'Effect of {threshold_name} on Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("features.csv")
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
    thresholds = manifest["thresholds_used"]

    print("=== Evaluating without smoothing ===")
    results = evaluate_classifier(df, thresholds, smoothed=False)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    for cls, acc in results['class_accuracies'].items():
        print(f"  {cls}: {acc['accuracy_pct']:.2f}% ({acc['n']} samples)")
    plot_confusion_matrix(results, "Confusion Matrix (Raw)")

    print("\n=== Evaluating with smoothing (3 of 5) ===")
    results_smoothed = evaluate_classifier(df, thresholds, window=5, vote=3, smoothed=True)
    print(f"Overall Accuracy: {results_smoothed['overall_accuracy']:.2f}%")
    for cls, acc in results_smoothed['class_accuracies'].items():
        print(f"  {cls}: {acc['accuracy_pct']:.2f}% ({acc['n']} samples)")
    plot_confusion_matrix(results_smoothed, "Confusion Matrix (Smoothed)")

    print("\n=== Running threshold experiment ===")
    experiment_results = experiment_threshold(
        df, thresholds,
        'night_brightness',
        thresholds['night_brightness'],
        range_pct=20,
        num_steps=5,
    )
    for r in experiment_results:
        print(f"  Threshold: {r['threshold_value']:.2f}, Accuracy: {r['overall_accuracy']:.2f}%")
    plot_experiment_results(experiment_results, 'night_brightness')