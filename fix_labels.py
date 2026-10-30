"""Фильтрует разметку: оставляет только правдоподобные ad_close (крестики).

Крестик рекламы: маленький, почти квадратный (0.6 < w/h < 1.8),
размер ~30-130 пикселей, максимум 2 на кадр.

Запуск: python fix_labels.py
"""
from pathlib import Path

IMG_DIR = Path("screenshorts")


def main():
    total_before = 0
    total_after = 0
    frames_with_ad = 0

    for label_path in sorted(IMG_DIR.glob("*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        kept = []
        ad_lines = []
        other_lines = []

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            if cls == 1:  # ad_close
                ad_lines.append((cx, cy, bw, bh))
            else:
                other_lines.append(line)

        total_before += len(ad_lines)

        # фильтруем крестики
        good = []
        for cx, cy, bw, bh in ad_lines:
            w_px = bw * 1920  # нормализуем к экрану 1920x1080
            h_px = bh * 1080
            if not (25 <= w_px <= 150 and 25 <= h_px <= 150):
                continue
            aspect = w_px / h_px if h_px > 0 else 0
            if not (0.5 <= aspect <= 2.0):
                continue
            good.append((cx, cy, bw, bh))

        # максимум 2 лучших (по размеру, близкие к эталону ~50px)
        good.sort(key=lambda d: abs(d[2] * 1920 - 55) + abs(d[3] * 1080 - 55))
        good = good[:2]

        for cx, cy, bw, bh in good:
            kept.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_after += 1

        if good:
            frames_with_ad += 1

        # перезаписываем только если были изменения
        if kept != ad_lines and (kept or other_lines):
            all_lines = other_lines + kept
            label_path.write_text("\n".join(all_lines), encoding="utf-8")

    print(f"ad_close до фильтра: {total_before}")
    print(f"ad_close после фильтра: {total_after}")
    print(f"кадров с ad_close: {frames_with_ad}")


if __name__ == "__main__":
    main()