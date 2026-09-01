from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import urllib.request
import io
import os
import random


def load_random_image(w=640, h=360):
    urls = [
        "https://picsum.photos/id/1018/800/600",
        "https://picsum.photos/id/1015/800/600",
        "https://picsum.photos/id/1043/800/600",
        "https://picsum.photos/id/1019/800/600",
        "https://picsum.photos/id/1035/800/600",
        "https://picsum.photos/id/1041/800/600",
        "https://picsum.photos/id/1059/800/600",
        "https://picsum.photos/id/1067/800/600",
        "https://picsum.photos/id/1077/800/600",
        "https://picsum.photos/id/1080/800/600",
        "https://picsum.photos/id/1084/800/600",
    ]
    random.shuffle(urls)
    for url in urls[:3]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((w, h))
            return np.asarray(img).astype(np.uint8)
        except Exception:
            continue
    base = np.random.RandomState(0).randint(80, 180, (h, w, 3), dtype=np.uint8)
    for _ in range(50):
        cy = random.randint(0, h)
        cx = random.randint(0, w)
        r = random.randint(20, 80)
        cv2_style = np.zeros_like(base)
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        base[mask] = base[mask] * 0.6 + np.array([180, 180, 150]) * 0.4
    return base


def synth_smoke_strong(seed):
    np.random.seed(seed)
    random.seed(seed)
    base = load_random_image()
    img = base.astype(np.float32)
    h, w = img.shape[:2]

    blurred_img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(7))
    blurred = np.array(blurred_img).astype(np.float32)

    base_color = 100 + np.random.randint(-15, 15)
    gray_overlay = np.full_like(img, base_color, dtype=np.float32)

    noise = np.random.normal(0, 18, (h, w)).astype(np.uint8)
    density_img = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(35))
    density_raw = np.array(density_img).astype(np.float32)
    dm = density_raw - density_raw.min()
    if dm.max() > 0:
        dm = dm / dm.max()

    density = np.random.uniform(0.80, 0.92)
    out = blurred * (1 - density * dm[..., None]) + gray_overlay * (density * dm[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def synth_rain_strong(seed):
    """Капли дождя — диагональные штрихи под ~45°.
    Даёт diag_ratio > 1, потому что оба градиента (gx, gy) большие и соразмерные,
    причём сумма |gy| доминирует над суммой |gx| благодаря длинным наклонным линиям.
    """
    np.random.seed(seed)
    random.seed(seed)
    base = load_random_image().astype(np.float32)
    h, w = base.shape[:2]

    n_drops = np.random.randint(4000, 7000)
    result = base.copy()

    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        # Длинные полосы — 50-90 пикселей длины
        length = np.random.randint(50, 90)
        # ГОРИЗОНТАЛЬНЫЕ штрихи (угол 0-15° от горизонтали).
        # Логика: горизонтальная линия имеет ДЛИННЫЙ вертикальный пробег градиента
        # на концах (по всей длине штриха, по 2 концам). gx — только на боковых краях.
        # → sum(|gy|) >> sum(|gx|) → diag_ratio > 1
        angle_deg = np.random.uniform(0, 15)
        brightness = np.random.uniform(210, 255)

        rad = np.deg2rad(angle_deg)
        dx = np.cos(rad) * length
        dy = np.sin(rad) * length
        steps = int(max(abs(dx), abs(dy))) + 1
        for s in range(steps + 1):
            t = s / max(steps, 1)
            xi = int(x + t * dx)
            yi = int(y + t * dy)
            if 0 <= xi < w and 0 <= yi < h:
                for channel in range(3):
                    result[yi, xi, channel] = (
                        0.15 * result[yi, xi, channel] + 0.85 * brightness
                    )

    result_pil = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    result_pil = result_pil.filter(ImageFilter.GaussianBlur(1))
    return np.array(result_pil).astype(np.uint8)


def synth_fog_strong(seed):
    np.random.seed(seed)
    random.seed(seed)
    base = load_random_image()
    h, w = base.shape[:2]

    blurred_img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(11))
    blurred = np.array(blurred_img).astype(np.float32)

    white = np.full_like(blurred, 220 + np.random.randint(-10, 10), dtype=np.float32)
    noise = np.random.normal(0, 8, (h, w)).astype(np.uint8)
    density_img = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(45))
    dm = np.array(density_img).astype(np.float32)
    dm = (dm - dm.min())
    if dm.max() > 0:
        dm = dm / dm.max()

    fog_amount = np.random.uniform(0.55, 0.75)
    out = blurred * (1 - fog_amount) + white * fog_amount
    out = out + dm[..., None] * 15
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def synth_night_strong(seed):
    np.random.seed(seed)
    random.seed(seed)
    base = load_random_image()
    pil = Image.fromarray(base)
    pil = ImageEnhance.Brightness(pil).enhance(0.18)
    pil = ImageEnhance.Contrast(pil).enhance(0.85)
    arr = np.array(pil).astype(np.float32)
    noise = np.random.normal(0, 4, arr.shape).astype(np.float32)
    out = arr + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def synth_off(seed):
    np.random.seed(seed)
    random.seed(seed)
    return load_random_image()


GENERATORS = {
    "OFF": synth_off,
    "NIGHT": synth_night_strong,
    "FOG": synth_fog_strong,
    "SMOKE": synth_smoke_strong,
    "RAIN": synth_rain_strong,
}


def main(out_root, n_per_class=12):
    os.makedirs(out_root, exist_ok=True)
    for cls, gen in GENERATORS.items():
        cls_dir = os.path.join(out_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            seed = hash((cls, i)) % (2 ** 31)
            arr = gen(seed)
            path = os.path.join(cls_dir, f"{cls.lower()}_{i+1:02d}.jpg")
            Image.fromarray(arr).save(path, "JPEG", quality=92)
        print(f"  {cls}: {n_per_class} frames")
    print(f"\nDone -> {out_root}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Никита\Desktop\pract123\test_dataset"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(target, n)