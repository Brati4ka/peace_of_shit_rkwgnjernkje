import sys
sys.path.insert(0, r'C:\Users\Никита\Desktop\pract123')
import pandas as pd
import numpy as np
from collections import deque, Counter

df = pd.read_csv('features.csv')
features = df[['brightness','std','laplac_var','edge_density','diag_ratio']].values
labels = df['class'].values


def classify_batch(features, labels, t):
    correct = 0
    for f, true_l in zip(features, labels):
        b, s, lv, ed, dr = f
        if b < t['night_b']: pred = 'NIGHT'
        elif lv <= t['fog_lv'] and ed <= t['fog_ed']: pred = 'FOG'
        elif t['smoke_b_lo'] <= b <= t['smoke_b_hi'] and t['smoke_s_lo'] <= s <= t['smoke_s_hi'] and t['smoke_lv_lo'] <= lv <= t['smoke_lv_hi'] and t['smoke_ed_lo'] <= ed <= t['smoke_ed_hi']: pred = 'SMOKE'
        elif ed >= t['rain_ed'] and dr <= t['rain_dr']: pred = 'RAIN'
        else: pred = 'OFF'
        if pred == true_l: correct += 1
    return correct / len(labels)


best = 0
best_t = None
for night_b in [70, 80, 90]:
    for fog_lv in [400, 800, 1500]:
        for fog_ed in [0.15, 0.25, 0.35]:
            for smoke_b_lo in [60, 80]:
                for smoke_b_hi in [150, 180]:
                    for smoke_lv_lo in [500, 1500]:
                        for smoke_lv_hi in [6000, 9000]:
                            for smoke_ed_lo in [0.05, 0.10]:
                                for smoke_ed_hi in [0.35, 0.50]:
                                    for rain_ed in [0.13, 0.18]:
                                        for rain_dr in [0.60, 0.80]:
                                            t = dict(
                                                night_b=night_b, fog_lv=fog_lv, fog_ed=fog_ed,
                                                smoke_b_lo=smoke_b_lo, smoke_b_hi=smoke_b_hi,
                                                smoke_s_lo=20, smoke_s_hi=90,
                                                smoke_lv_lo=smoke_lv_lo, smoke_lv_hi=smoke_lv_hi,
                                                smoke_ed_lo=smoke_ed_lo, smoke_ed_hi=smoke_ed_hi,
                                                rain_ed=rain_ed, rain_dr=rain_dr,
                                            )
                                            acc = classify_batch(features, labels, t)
                                            if acc > best:
                                                best = acc
                                                best_t = t
print(f'Best acc: {best*100:.2f}%')
print(f'Best params: {best_t}')