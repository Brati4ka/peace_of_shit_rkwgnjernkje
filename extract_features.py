# Извлекает 5 статистических признаков (brightness, std, laplac_var, edge_density, diag_ratio) из всех изображений 5-классового датасета (OFF/NIGHT/FOG/SMOKE/RAIN) и сохраняет в features.csv.

import cv2
import numpy as np
import csv
from pathlib import Path

"""Extract features from a BGR image."""
def extract_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    brightness = float(gray.mean())
    std = float(gray.std())
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplac_var = float(laplacian.var())

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)
    edge_density = float((grad > 50).mean())

    ax = float(np.abs(sx).sum())
    ay = float(np.abs(sy).sum())
    diag_ratio = ay / ax if ax > 1e-6 else 1.0

    if gray.size == 0 or brightness < 1.0:
        return None

    return brightness, std, laplac_var, edge_density, diag_ratio


def process_dataset(dataset_path, output_csv):
    dataset_path = Path(dataset_path)
    classes = ['OFF', 'NIGHT', 'FOG', 'SMOKE', 'RAIN']

    skipped_total = 0

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class', 'brightness', 'std', 'laplac_var',
                         'edge_density', 'diag_ratio', 'filename'])

        for class_name in classes:
            class_dir = dataset_path / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue

            image_files = sorted(
                list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            )
            print(f"Processing {class_name}: {len(image_files)} images")

            skipped_class = 0
            written_class = 0

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Warning: Could not read {img_path}, skipping")
                        skipped_class += 1
                        continue

                    if min(img.shape[:2]) < 32:
                        print(f"Warning: {img_path} too small, skipping")
                        skipped_class += 1
                        continue

                    features = extract_features(img)
                    if features is None:
                        print(f"Warning: {img_path} produced empty/black frame, skipping")
                        skipped_class += 1
                        continue

                    writer.writerow([class_name] + list(features) + [img_path.name])
                    written_class += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    skipped_class += 1

            skipped_total += skipped_class
            print(f"  -> written {written_class}, skipped {skipped_class}")

    print(f"\nFeatures saved to {output_csv}; total skipped {skipped_total}")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "pract"
    process_dataset(src, "features.csv")