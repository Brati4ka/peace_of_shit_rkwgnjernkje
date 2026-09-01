from PIL import Image
import numpy as np
import os, sys, json
sys.path.insert(0, r'C:\Users\Никита\Desktop\pract123')


def extract_features_rgb(arr):
    arr_bgr = arr[..., ::-1].copy()
    gray = 0.299 * arr_bgr[..., 2] + 0.587 * arr_bgr[..., 1] + 0.114 * arr_bgr[..., 0]
    brightness = float(gray.mean())
    std = float(gray.std())
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    laplac = np.zeros_like(gray)
    laplac[1:-1, 1:-1] = (
        gray[1:-1, :-2] + gray[1:-1, 2:] +
        gray[:-2, 1:-1] + gray[2:, 1:-1] -
        4 * gray[1:-1, 1:-1]
    )
    laplac_var = float(laplac.var())
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((grad > 50).mean())
    ax = float(np.abs(gx).sum())
    ay = float(np.abs(gy).sum())
    diag_ratio = ay / ax if ax > 1e-6 else 1.0
    return brightness, std, laplac_var, edge_density, diag_ratio


def calibrate(values_a, values_b, default=None):
    if len(values_a) < 2 or len(values_b) < 2:
        return default
    p95_a = np.percentile(values_a, 95)
    p5_b = np.percentile(values_b, 5)
    if p95_a < p5_b:
        return float((p95_a + p5_b) / 2.0)
    return default


def main(data_root):
    classes = ['OFF', 'NIGHT', 'FOG', 'SMOKE', 'RAIN']
    rows = []
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith(('.jpg', '.png')):
                continue
            try:
                arr = np.asarray(Image.open(os.path.join(cls_dir, fn)).convert('RGB'))
            except Exception:
                continue
            f = extract_features_rgb(arr)
            rows.append((cls, *f))
    print(f'Loaded {len(rows)} rows')

    df = {}
    for c in classes:
        df[c] = [r for r in rows if r[0] == c]
    for c in classes:
        print(f'  {c}: {len(df[c])}')

    def arr(c, idx):
        return np.array([r[idx] for r in df[c]])

    t = {}
    t['night_brightness'] = calibrate(arr('NIGHT', 1), arr('OFF', 1), default=55.0)
    t['fog_brightness_min'] = calibrate(np.concatenate([arr('OFF', 1), arr('NIGHT', 1), arr('SMOKE', 1)]),
                                        arr('FOG', 1), default=150.0)
    t['fog_std_max'] = calibrate(np.concatenate([arr('OFF', 2), arr('NIGHT', 2), arr('RAIN', 2)]),
                                 arr('FOG', 2), default=40.0)
    t['fog_laplac_max'] = calibrate(np.concatenate([arr('OFF', 3), arr('NIGHT', 3), arr('RAIN', 3)]),
                                    arr('FOG', 3), default=100.0)
    t['smoke_brightness_range'] = [
        calibrate(arr('NIGHT', 1), arr('SMOKE', 1), default=60.0),
        calibrate(arr('SMOKE', 1), arr('FOG', 1), default=150.0)
    ]
    if t['smoke_brightness_range'][0] >= t['smoke_brightness_range'][1]:
        t['smoke_brightness_range'] = [60.0, 150.0]
    t['smoke_std_max'] = calibrate(np.concatenate([arr('OFF', 2), arr('FOG', 2), arr('NIGHT', 2)]),
                                   arr('SMOKE', 2), default=30.0)
    t['smoke_laplac_max'] = calibrate(np.concatenate([arr('OFF', 3), arr('NIGHT', 3), arr('RAIN', 3)]),
                                      arr('SMOKE', 3), default=150.0)
    t['rain_edge_density_min'] = calibrate(np.concatenate([arr('OFF', 4), arr('NIGHT', 4), arr('FOG', 4), arr('SMOKE', 4)]),
                                           arr('RAIN', 4), default=0.04)
    t['rain_diagonal_ratio_min'] = calibrate(np.concatenate([arr('OFF', 5), arr('NIGHT', 5), arr('FOG', 5), arr('SMOKE', 5)]),
                                              arr('RAIN', 5), default=1.30)

    print()
    print('=== Calibrated thresholds ===')
    print(json.dumps(t, indent=2))

    def classify(features, t):
        b, s, lv, ed, dr = features
        if b < t["night_brightness"]:
            return "NIGHT"
        if (b >= t["fog_brightness_min"] and s <= t["fog_std_max"] and lv <= t["fog_laplac_max"]):
            return "FOG"
        sb_lo, sb_hi = t["smoke_brightness_range"]
        if (sb_lo <= b <= sb_hi and s <= t["smoke_std_max"] and lv <= t["smoke_laplac_max"]):
            return "SMOKE"
        if (ed >= t["rain_edge_density_min"] and dr >= t["rain_diagonal_ratio_min"]):
            return "RAIN"
        return "OFF"

    from collections import Counter
    cm = {c: Counter() for c in classes}
    correct_per_class = Counter()
    total_per_class = Counter()
    for cls, *f in rows:
        pred = classify(f, t)
        cm[cls][pred] += 1
        total_per_class[cls] += 1
        if pred == cls:
            correct_per_class[cls] += 1

    print()
    print('=== Per-class accuracy ===')
    overall_correct = 0
    for c in classes:
        n = total_per_class[c]
        k = correct_per_class[c]
        acc = 100.0 * k / n if n > 0 else 0
        overall_correct += k
        print(f'  {c}: {k}/{n} = {acc:.1f}%')
    overall = 100.0 * overall_correct / len(rows) if rows else 0
    print(f'\n=== Overall accuracy: {overall:.1f}% ===')

    print()
    print('=== Confusion matrix (rows=true, cols=pred) ===')
    header = 'true\\pred'.ljust(12) + ''.join(c.ljust(8) for c in classes)
    print(header)
    for c in classes:
        row = c.ljust(12) + ''.join(str(cm[c][p]).ljust(8) for p in classes)
        print(row)

    out_path = r'C:\Users\Никита\Desktop\pract123\manifest_test.json'
    with open(out_path, 'w') as f:
        json.dump(t, f, indent=2)
    print(f'\nSaved -> {out_path}')

    return t


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Никита\Desktop\pract123\test_dataset"
    main(target)