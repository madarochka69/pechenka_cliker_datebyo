"""Авторазметчик Cookie Clicker: печенька + реклама + улучшения.

Использует цветовые диапазоны + позицию на экране + размер.
Запуск: python color_label.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}
CLASSES = ['game', 'ad_close', 'upgrade']
COLORS = [(0, 200, 0), (0, 0, 255), (255, 120, 0)]


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    buf.tofile(str(path))


def find_game(hsv, h, w):
    """Печенька: большая золотая в левой половине экрана."""
    # золотистый диапазон
    lower = np.array([10, 60, 120])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # морфология
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ищем только в левой половине (печенька слева)
    left_region = mask[:, :w//2]
    contours, _ = cv2.findContours(left_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20000 or area > 200000:  # печенька ~70000-90000px на 1920x1080
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # проверяем что не слишком вытянутая
        if bw / bh > 2.0 or bh / bw > 2.0:
            continue
        dets.append((x, y, x + bw, y + bh))
        break  # только одна печенька
    return dets


def find_ad_close(hsv, h, w):
    """Крестик рекламы: маленький яркий элемент в углах."""
    # яркие белые/серые пиксели
    lower = np.array([0, 0, 220])
    upper = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # реклама обычно в правом нижнем углу или левом верхнем
    # проверим только угловые области
    margin_x, margin_y = int(w * 0.3), int(h * 0.3)
    
    # 4 угла
    corners = [
        mask[:margin_y, :margin_x],                    # TL
        mask[:margin_y, w-margin_x:w],                  # TR
        mask[h-margin_y:h, :margin_x],                  # BL
        mask[h-margin_y:h, w-margin_x:w],               # BR
    ]
    offsets = [(0, 0), (w-margin_x, 0), (0, h-margin_y), (w-margin_x, h-margin_y)]
    
    dets = []
    for corner, (ox, oy) in zip(corners, offsets):
        kernel = np.ones((3, 3), np.uint8)
        corner = cv2.morphologyEx(corner, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(corner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:  # крестик ~500-2000px
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw / bh > 3.0 or bh / bw > 3.0:
                continue
            dets.append((ox + x, oy + y, ox + x + bw, oy + y + bh))
    return dets


def find_upgrade(hsv, h, w):
    """Улучшения: золотые здания в правой части экрана."""
    lower = np.array([12, 80, 100])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # магазин в правой половине
    right_region = mask[:, 3*w//4:]
    offset_x = 3*w//4

    kernel = np.ones((5, 5), np.uint8)
    right_region = cv2.morphologyEx(right_region, cv2.MORPH_CLOSE, kernel)
    right_region = cv2.morphologyEx(right_region, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(right_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # объединяем всё одной большой рамкой
        dets.append((offset_x + x, y, offset_x + x + bw, y + bh))
    return dets


def draw_preview(img, all_dets, out_path, max_width=1280):
    vis = img.copy()
    for cls_name, dets in all_dets.items():
        idx = CLASSES.index(cls_name)
        col = COLORS[idx % len(COLORS)]
        for x1, y1, x2, y2 in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, max(2, img.shape[1] // 600))
            cv2.putText(vis, cls_name, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 1800), col, 2)
    if vis.shape[1] > max_width:
        vis = cv2.resize(vis, None, fx=max_width / vis.shape[1], fy=max_width / vis.shape[1])
    write_image(out_path, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="screenshorts")
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(f"Папка {images_dir} не найдена")

    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"Нет картинок в {images_dir}")

    counts = {c: 0 for c in CLASSES}
    total = len(files)

    for i, path in enumerate(files, 1):
        img = read_image(path)
        if img is None:
            print(f"[!] не удалось открыть {path.name}")
            continue

        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        all_dets = {}
        label_lines = []

        # game
        dets = find_game(hsv, h, w)
        all_dets['game'] = dets
        for x1, y1, x2, y2 in dets:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts['game'] += 1

        # ad_close
        dets = find_ad_close(hsv, h, w)
        all_dets['ad_close'] = dets
        for x1, y1, x2, y2 in dets:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts['ad_close'] += 1

        # upgrade
        dets = find_upgrade(hsv, h, w)
        all_dets['upgrade'] = dets
        if dets:
            # объединяем все рамки улучшений в одну большую
            xs = [x1 for x1, _, _, _ in dets] + [x3 for _, _, x3, _ in dets]
            ys = [y1 for _, y1, _, _ in dets] + [y4 for _, _, _, y4 in dets]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts['upgrade'] += 1
            all_dets['upgrade'] = [(x1, y1, x2, y2)]

        if label_lines:
            label_path = images_dir / (path.stem + ".txt")
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if args.preview:
            draw_preview(img, all_dets, Path(args.preview) / (path.stem + ".jpg"))

        n_found = sum(len(v) for v in all_dets.values())
        print(f"[{i}/{total}] {path.name}: {n_found} объектов "
              f"(game={len(all_dets['game'])}, ad_close={len(all_dets['ad_close'])}, upgrade={len(all_dets['upgrade'])})")

    print("\n=== Итог ===")
    print(f"Размечено: {total} кадров")
    for c, n in counts.items():
        print(f"  {c}: {n} рамок")
    if args.preview:
        print(f"Проверь '{args.preview}/' — зелёные=game, красные=ad_close, оранжевые=upgrade")


if __name__ == "__main__":
    main()