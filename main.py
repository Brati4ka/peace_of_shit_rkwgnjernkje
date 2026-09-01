# Сквозной пайплайн: извлечение признаков -> гистограммы -> калибровка порогов -> оценка классификатора -> manifest + матрица ошибок -> эксперимент с порогом.

import pandas as pd

from extract_features import process_dataset
from analyze_distributions import (load_features, plot_all_histograms,
                                   create_separation_table)
from calibrate_thresholds import calibrate_all_thresholds, save_manifest
from classify import (evaluate_classifier, plot_confusion_matrix,
                      experiment_threshold, plot_experiment_results)


def main():
    print("=" * 60)
    print("SCENE CLASSIFIER PRACTICAL WORK")
    print("=" * 60)

    print("\n[Step 1] Extracting features from dataset...")
    process_dataset("pract/", "features.csv")

    print("\n[Step 2] Analyzing feature distributions...")
    df = load_features("features.csv")
    plot_all_histograms(df)
    create_separation_table(df)

    print("\n[Step 3] Calibrating thresholds...")
    thresholds = calibrate_all_thresholds(df)

    print("\n[Step 4] Evaluating classifier (raw)...")
    results = evaluate_classifier(df, thresholds, smoothed=False)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")

    print("\n[Step 5] Saving manifest...")
    save_manifest(thresholds, {
        'overall': results['overall_accuracy'],
        'by_mode': results['class_accuracies']
    })

    print("\n[Step 6] Plotting confusion matrix...")
    plot_confusion_matrix(results, "Confusion Matrix")

    print("\n[Step 7] Running threshold experiment...")
    threshold_to_vary = 'night_brightness'
    base_value = thresholds[threshold_to_vary]

    experiment_payload = experiment_threshold(
        df, thresholds,
        threshold_to_vary,
        base_value,
        range_pct=20,
        num_steps=5,
    )

    exp_results = experiment_payload['results']
    exp_label = experiment_payload['label']

    print(f"\n{'Threshold':<15} {'Accuracy (%)':<15}")
    print("-" * 30)
    for r in exp_results:
        print(f"{r['threshold_value']:<15.2f} {r['overall_accuracy']:<15.2f}")

    plot_experiment_results(experiment_payload)

    print("\n[Step 8] Saving experiment results...")
    pd.DataFrame([{
        'threshold': r['threshold_value'],
        'accuracy': r['overall_accuracy']
    } for r in exp_results]).to_csv('experiment_results.csv', index=False)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)
    print("Generated files:")
    print("  - features.csv")
    print("  - manifest.json")
    print("  - confusion_matrix.png")
    print("  - experiment_results.csv")
    print("  - experiment_results.png")
    print("  - plots/")


if __name__ == "__main__":
    main()