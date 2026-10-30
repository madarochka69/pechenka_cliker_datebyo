"""Ищет скриншоты, где вероятно есть реклама (яркий крестик в углу экрана).

Выводит список файлов + позиции ярких областей. Помогает выбрать скриншот
для crop_template.py ad_close.
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
    if not images_dir.exists():
        sys.exit("Папка screenshorts не найдена")

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    print("Скриншоты с вероятной рекламой (яркие области в углах):\n")

    found_any = False
    for path in files:
        img = read_image(path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # яркие области (порог высокий — почти белые)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # рассматриваем только углы экрана (реклама обычно в углу)
        margin_x, margin_y = int(w * 0.25), int(h * 0.25)
        corners = {
            "TL": mask[0:margin_y, 0:margin_x],
            "TR": mask[0:margin_y, w - margin_x:w],
            "BL": mask[h - margin_y:h, 0:margin_x],
            "BR": mask[h - margin_y:h, w - margin_x:w],
        }

        bright = []
        for name, corner in corners.items():
            n = int(corner.sum() / 255)
            if n > 200:  # заметное белое пятно в углу
                bright.append(f"{name}:{n}")

        if bright:
            found_any = True
            print(f"  {path.name}: {', '.join(bright)}")

    if not found_any:
        print("  Не найдено скриншотов с явными яркими углами.")
        print("  Возможно, рекламы вообще нет на этих скриншотах.")
        print("  Тогда можно: либо обойтись без класса ad_close,")
        print("  либо открыть скриншоты вручную и найти рекламу.")


if __name__ == "__main__":
    main()