"""Собирает коллаж из скриншотов в один файл для быстрой проверки.

Запуск: python montage.py
Создаёт montage.png — сетку из первых N скриншотов.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}
COLS = 4       # картинок в ряд
CELL_W = 320   # ширина одной ячейки
MAX_ROWS = 4   # сколько рядов


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def main():
    images_dir = Path("screenshorts")
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit("Нет скриншотов")

    files = files[: COLS * MAX_ROWS]
    rows = []
    for r in range(MAX_ROWS):
        row_imgs = []
        for c in range(COLS):
            i = r * COLS + c
            if i >= len(files):
                break
            img = read_image(files[i])
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = CELL_W / w
            img = cv2.resize(img, (CELL_W, int(h * scale)))
            # подпись имени файла
            cv2.putText(img, files[i].stem[-12:], (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            row_imgs.append(img)
        if row_imgs:
            h_max = max(i.shape[0] for i in row_imgs)
            row_imgs = [cv2.copyMakeBorder(i, 0, h_max - i.shape[0], 0, 0,
                                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
                        for i in row_imgs]
            rows.append(np.hstack(row_imgs))

    if not rows:
        sys.exit("Не удалось прочитать скриншоты")
    h_max = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, h_max - r.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for r in rows]
    montage = np.vstack(rows)

    out = Path("montage.png")
    ok, buf = cv2.imencode(".png", montage)
    buf.tofile(str(out))
    print(f"Сохранено: {out} ({montage.shape[1]}x{montage.shape[0]})")
    print(f"Показаны файлы: {', '.join(f.stem for f in files)}")


if __name__ == "__main__":
    main()