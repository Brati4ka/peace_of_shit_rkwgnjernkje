# Гистограммы распределения признаков по классам, p5/p95-статистики и сводная таблица разделимости (аналог Таблицы 5 из ТЗ) на основе features.csv.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FEATURES = ['brightness', 'std', 'laplac_var', 'edge_density', 'diag_ratio']
FEATURE_BINS = {'brightness': 30, 'std': 30, 'laplac_var': 30,
                'edge_density': 25, 'diag_ratio': 30}
CLASS_COLORS = {'OFF': 'tab:blue', 'NIGHT': 'tab:red', 'FOG': 'tab:green',
                'SMOKE': 'tab:orange', 'RAIN': 'tab:purple'}

"""Analyze feature distributions across different classes."""

def load_features(csv_file):
    return pd.read_csv(csv_file)


def plot_all_histograms(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    classes = sorted(df['class'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        bins = FEATURE_BINS[feature]
        for class_name in classes:
            data = df[df['class'] == class_name][feature].dropna()
            if len(data) == 0:
                continue
            ax.hist(data, bins=bins, alpha=0.5,
                    label=class_name,
                    color=CLASS_COLORS.get(class_name, 'gray'),
                    density=True)
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=150)
    plt.close()

    for feature in FEATURES:
        plt.figure(figsize=(10, 6))
        bins = FEATURE_BINS[feature]
        for class_name in classes:
            data = df[df['class'] == class_name][feature].dropna()
            if len(data) == 0:
                continue
            plt.hist(data, bins=bins, alpha=0.5,
                     label=class_name,
                     color=CLASS_COLORS.get(class_name, 'gray'),
                     density=True)
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_distribution.png", dpi=150)
        plt.close()


def analyze_separation(df):
    classes = sorted(df['class'].unique())
    separation_analysis = {}

    for feature in FEATURES:
        print(f"\n=== Analysis for {feature} ===")
        print("Class\tMean\tStd\tMin\tMax\tp5\tp95")
        print("-" * 70)

        class_stats = {}
        for class_name in classes:
            data = df[df['class'] == class_name][feature].dropna()
            if len(data) == 0:
                continue
            stats = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'p5': float(data.quantile(0.05)),
                'p95': float(data.quantile(0.95)),
            }
            class_stats[class_name] = stats
            print(f"{class_name}\t{stats['mean']:.2f}\t{stats['std']:.2f}\t"
                  f"{stats['min']:.2f}\t{stats['max']:.2f}\t"
                  f"{stats['p5']:.2f}\t{stats['p95']:.2f}")

        print("\nSafe zones:")
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                if c1 not in class_stats or c2 not in class_stats:
                    continue
                p95_1, p5_2 = class_stats[c1]['p95'], class_stats[c2]['p5']
                p95_2, p5_1 = class_stats[c2]['p95'], class_stats[c1]['p5']
                if p95_1 < p5_2:
                    print(f"  {c1} vs {c2}: Safe zone [{p95_1:.2f}, {p5_2:.2f}], "
                          f"midpoint = {(p95_1 + p5_2) / 2:.2f}")
                elif p95_2 < p5_1:
                    print(f"  {c2} vs {c1}: Safe zone [{p95_2:.2f}, {p5_1:.2f}], "
                          f"midpoint = {(p95_2 + p5_1) / 2:.2f}")
                else:
                    print(f"  {c1} vs {c2}: NO safe zone - distributions overlap")

        separation_analysis[feature] = class_stats

    return separation_analysis


def create_separation_table(df):
    classes = sorted(df['class'].unique())

    print("\n" + "=" * 70)
    print("TABLE 5: Class Separability by Features")
    print("=" * 70)
    print(f"{'Feature':<15} {'Good separation':<35} {'Poor separation':<20}")
    print("-" * 90)

    for feature in FEATURES:
        good_sep, poor_sep = [], []
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                d1 = df[df['class'] == c1][feature].dropna()
                d2 = df[df['class'] == c2][feature].dropna()
                if len(d1) == 0 or len(d2) == 0:
                    continue
                p95_1, p5_2 = d1.quantile(0.95), d2.quantile(0.05)
                p95_2, p5_1 = d2.quantile(0.95), d1.quantile(0.05)
                if (p95_1 < p5_2) or (p95_2 < p5_1):
                    good_sep.append(f"{c1}/{c2}")
                else:
                    poor_sep.append(f"{c1}/{c2}")
        print(f"{feature:<15} "
              f"{', '.join(good_sep) if good_sep else 'None':<35} "
              f"{', '.join(poor_sep) if poor_sep else 'None'}")


if __name__ == "__main__":
    df = load_features("features.csv")
    plot_all_histograms(df)
    analyze_separation(df)
    create_separation_table(df)