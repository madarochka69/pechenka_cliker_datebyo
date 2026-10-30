"""Анализирует яркость и разнообразие скриншотов.

Помогает понять: игра на переднем плане или экран тёмный/свёрнут.
Запуск: python screen_stats.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def main():
    images_dir = Path("screenshorts")
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit("Нет скриншотов")

    print(f"Файлов: {len(files)}\n")
    print(f"{'файл':<40} {'яркость':>8} {'цветность':>8} {'размер':>8}")

    brightness_all = []
    first_pair_diff = None
    prev = None

    for i, path in enumerate(files):
        img = read_image(path)
        if img is None:
            continue
        bgr = np.float32(img)
        brightness = bgr.mean()  # 0..255
        # цветность: среднее отклонение каналов друг от друга
        colorfulness = np.abs(bgr[:, :, 0] - bgr[:, :, 1]).mean() + \
            np.abs(bgr[:, :, 1] - bgr[:, :, 2]).mean()
        h, w = img.shape[:2]
        brightness_all.append(brightness)

        if i == 0:
            first_pair_diff = (brightness, colorfulness)
        print(f"{path.name:<40} {brightness:8.1f} {colorfulness:8.1f} {w}x{h}")

        # разница между соседними кадрами
        if prev is not None:
            diff = cv2.absdiff(img, prev)
            md = diff.mean()
            if i == 1:
                print(f"\nСредняя разница между 1-м и 2-м кадром: {md:.2f}")
        prev = img

    avg = np.mean(brightness_all)
    print(f"\nСредняя яркость всех кадров: {avg:.1f}")
    if avg < 60:
        print(">>> Экран ТЁМНЫЙ — игра, скорее всего, свёрнута или не на переднем плане!")
        print(">>> Нужно открыть игру на весь экран и запустить screener.py заново.")
    elif avg < 120:
        print(">>> Яркость средняя — возможно, частично перекрыта окнами.")
    else:
        print(">>> Яркость нормальная — игра видна на экране.")


if __name__ == "__main__":
    main()