# Калибрует 9 порогов дерева решений SceneAnalyzer методом "середины безопасной зоны" и сохраняет в manifest.json.

import pandas as pd
import numpy as np
import json

DEFAULT_THRESHOLDS = {
    "night_brightness": 55.0,
    "fog_brightness_min": 150.0,
    "fog_std_max": 40.0,
    "fog_laplac_max": 100.0,
    "smoke_brightness_range": [60.0, 150.0],
    "smoke_std_max": 30.0,
    "smoke_laplac_max": 150.0,
    "rain_edge_density_min": 0.04,
    "rain_diagonal_ratio_min": 1.30,
}

""" Calibrate thresholds for scene classification based on feature distributions.
This module provides functions to calibrate thresholds for different scene classes
based on the extracted features from a dataset. It calculates thresholds for night, fog, smoke, and rain classes using statistical methods and saves the results in a manifest file.
"""
def calibrate_threshold(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    p95_a = np.percentile(values_a, 95)
    p5_b = np.percentile(values_b, 5)
    if p95_a < p5_b:
        return float((p95_a + p5_b) / 2.0)
    return None


def _safe(default_key):
    return DEFAULT_THRESHOLDS[default_key]


def _ensure_min_lt_max(lo, hi, fallback_lo, fallback_hi):
    if lo >= hi:
        print(f"  WARNING: calibrated smoke brightness range invalid "
              f"(lo={lo:.2f} >= hi={hi:.2f}); using defaults "
              f"[{fallback_lo}, {fallback_hi}]")
        return float(fallback_lo), float(fallback_hi)
    return lo, hi


def calibrate_all_thresholds(df):
    thresholds = {}

    night_values = df[df['class'] == 'NIGHT']['brightness']
    other_values = df[df['class'] != 'NIGHT']['brightness']
    night_threshold = calibrate_threshold(night_values, other_values)
    if night_threshold is None:
        night_threshold = _safe("night_brightness")
    thresholds['night_brightness'] = night_threshold

    fog_brightness = df[df['class'] == 'FOG']['brightness']
    non_fog = df[df['class'] != 'FOG']['brightness']
    fog_brightness_min = calibrate_threshold(non_fog, fog_brightness)
    if fog_brightness_min is None:
        fog_brightness_min = _safe("fog_brightness_min")
    thresholds['fog_brightness_min'] = fog_brightness_min

    fog_std = df[df['class'] == 'FOG']['std']
    non_fog_std = df[df['class'] != 'FOG']['std']
    fog_std_max = calibrate_threshold(non_fog_std, fog_std)
    if fog_std_max is None:
        fog_std_max = _safe("fog_std_max")
    thresholds['fog_std_max'] = fog_std_max

    fog_laplac = df[df['class'] == 'FOG']['laplac_var']
    non_fog_laplac = df[df['class'] != 'FOG']['laplac_var']
    fog_laplac_max = calibrate_threshold(non_fog_laplac, fog_laplac)
    if fog_laplac_max is None:
        fog_laplac_max = _safe("fog_laplac_max")
    thresholds['fog_laplac_max'] = fog_laplac_max

    smoke_brightness = df[df['class'] == 'SMOKE']['brightness']
    night_b = df[df['class'] == 'NIGHT']['brightness']
    fog_b = df[df['class'] == 'FOG']['brightness']

    smoke_min = calibrate_threshold(night_b, smoke_brightness)
    if smoke_min is None:
        smoke_min = _safe("smoke_brightness_range")[0]
    smoke_max = calibrate_threshold(smoke_brightness, fog_b)
    if smoke_max is None:
        smoke_max = _safe("smoke_brightness_range")[1]
    smoke_min, smoke_max = _ensure_min_lt_max(
        smoke_min, smoke_max,
        fallback_lo=DEFAULT_THRESHOLDS["smoke_brightness_range"][0],
        fallback_hi=DEFAULT_THRESHOLDS["smoke_brightness_range"][1],
    )
    thresholds['smoke_brightness_range'] = [smoke_min, smoke_max]

    smoke_std = df[df['class'] == 'SMOKE']['std']
    non_smoke_std = df[df['class'] != 'SMOKE']['std']
    smoke_std_max = calibrate_threshold(non_smoke_std, smoke_std)
    if smoke_std_max is None:
        smoke_std_max = _safe("smoke_std_max")
    thresholds['smoke_std_max'] = smoke_std_max

    smoke_laplac = df[df['class'] == 'SMOKE']['laplac_var']
    non_smoke_laplac = df[df['class'] != 'SMOKE']['laplac_var']
    smoke_laplac_max = calibrate_threshold(non_smoke_laplac, smoke_laplac)
    if smoke_laplac_max is None:
        smoke_laplac_max = _safe("smoke_laplac_max")
    thresholds['smoke_laplac_max'] = smoke_laplac_max

    rain_edge = df[df['class'] == 'RAIN']['edge_density']
    non_rain_edge = df[df['class'] != 'RAIN']['edge_density']
    rain_edge_density_min = calibrate_threshold(non_rain_edge, rain_edge)
    if rain_edge_density_min is None:
        rain_edge_density_min = _safe("rain_edge_density_min")
    thresholds['rain_edge_density_min'] = rain_edge_density_min

    rain_diag = df[df['class'] == 'RAIN']['diag_ratio']
    non_rain_diag = df[df['class'] != 'RAIN']['diag_ratio']
    rain_diag_min = calibrate_threshold(non_rain_diag, rain_diag)
    if rain_diag_min is None:
        rain_diag_min = _safe("rain_diagonal_ratio_min")
    thresholds['rain_diagonal_ratio_min'] = rain_diag_min

    return thresholds


def save_manifest(thresholds, accuracy_data, filename="manifest.json"):
    manifest = {
        "thresholds_used": thresholds,
        "overall_auto_accuracy_pct": float(accuracy_data.get('overall', 0.0)),
        "by_mode": accuracy_data.get('by_mode', {})
    }
    with open(filename, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {filename}")


if __name__ == "__main__":
    df = pd.read_csv("features.csv")
    thresholds = calibrate_all_thresholds(df)
    print("\nCalibrated Thresholds:")
    print(json.dumps(thresholds, indent=2))
    save_manifest(thresholds, {'overall': 0.0, 'by_mode': {}})