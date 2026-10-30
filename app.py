"""Кликер-бот для Cookie Clicker.

Как работает:
  - Каждые ~1.5 секунды YOLO ищет печеньку (класс game) и сам наводит на неё
    курсор — вручную ставить курсор больше не нужно.
  - Кликает печеньку нативными WinAPI вызовами (ctypes) — курсор не дёргается.
  - Каждые 7 секунд уходит на апгрейды (правая колонка), проходит сверху вниз
    по всей высоте, затем возвращается на печеньку.

Запуск:
    python app.py

Аварийная остановка: Ctrl+C или мышь в угол экрана.
"""
import argparse
import ctypes
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  # иначе между командами pyautogui встроенная пауза 0.1с!

# WinAPI mouse constants
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
mouse_event = ctypes.windll.user32.mouse_event

# WinAPI keyboard constants (пауза/продолжение по пробелу)
VK_SPACE = 0x20
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState


def space_pressed():
    """True, если пробел сейчас зажат (WinAPI, без перехвата ввода)."""
    return bool(GetAsyncKeyState(VK_SPACE) & 0x8000)

# ---------- зона апгрейда ----------
UPGRADE_INTERVAL = 7.0  # каждые 7 сек — проход по апгрейдам, затем снова печенька
UPGRADE_X = 1600
UPGRADE_Y_START = 400
UPGRADE_Y_END = 1045  # до самого низа, чтобы добить нижние апгрейды
UPGRADE_STEP = 35  # плотнее, чтобы не пропускать ряды
UPGRADE_HOLD = 0.02
UPGRADE_LINE_DELAY = 0.06
UPGRADE_JITTER = 1
# Дефолтная зона апгрейдов (правый столбец) — если YOLO её не нашёл
UPGRADE_ZONE_DEFAULT = (UPGRADE_X - 90, 150, UPGRADE_X + 90, UPGRADE_Y_END)

COOKIE_CHECK_INTERVAL = 1.5  # переприцел на печеньку (YOLO)
CLICK_INTERVAL = 0.05  # 20 кликов/сек — плавно, без лагов

OLD_MODEL_PATH = '/Users/coteunik/Desktop/python_learn/cookie_clicker_data/yolo/runs/detect/cookie_detecror/weights/best.pt'


def find_model_path(cli_value):
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value))
    for base in (Path('.'), Path(__file__).parent):
        candidates += sorted(
            (base / 'runs' / 'detect').glob('*/weights/best.pt'),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        candidates += sorted(base.glob('cookie_detecror*/weights/best.pt'))
    candidates.append(Path(OLD_MODEL_PATH))
    for c in candidates:
        if c.is_file():
            return c
    searched = '\n  '.join(str(c) for c in candidates)
    sys.exit(f'Не найдена обученная модель. Искал:\n  {searched}\n'
             f'Сначала обучите её: python train.py, либо укажите --model.')


def win_click():
    """Клик на WinAPI — курсор НЕ шевелится, нет задержек pyautogui."""
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.008)
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


class CookieBot:
    def __init__(self, model_path, conf):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.scale = pyautogui.screenshot().width / pyautogui.size().width
        self.last_upgrade_time = 0.0
        self.last_detect_time = 0.0
        self.stats = {'cookie': 0, 'upgrade': 0}
        self.cookie_pos = None  # (x, y) — куда сейчас целится бот
        self.cookie_target = None  # центр cookie (x, y) для расчёта дельты
        # зона апгрейдов: по умолчанию — правый столбец, YOLO её уточняет
        self.upgrade_zone = UPGRADE_ZONE_DEFAULT
        self.upgrade_zone_found = False  # был ли прямоугольник реально найден YOLO
        self.paused = False  # пауза по пробелу
        self._space_up = True  # чтобы пробел срабатывал по нажатию, а не удержанию

    def detect(self):
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        results = self.model(frame, conf=self.conf, verbose=False)
        found = {'game': [], 'upgrade': []}
        for result in results:
            for box in result.boxes:
                name = self.model.names[int(box.cls[0])]
                if name not in found:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                found[name].append((
                    conf_val,
                    (x1 + x2) / 2 / self.scale,
                    (y1 + y2) / 2 / self.scale,
                    (x2 - x1) / self.scale,
                    (y2 - y1) / self.scale,
                ))
        return found

    def aim_cookie(self):
        """Навести курсор на печеньку (с небольшим джиттером)."""
        if self.cookie_target is None:
            return False
        cx, cy = self.cookie_target
        jx = cx + random.randint(-4, 4)
        jy = cy + random.randint(-4, 4)
        self.cookie_pos = (jx, jy)
        pyautogui.moveTo(jx, jy, duration=0.05)
        return True

    def handle_detection(self, found):
        """Обработать результаты YOLO: печенька/апгрейды -> обновить цель."""
        self.upgrade_zone_found = False  # сбрасываем: зона действительна только для текущего кадра
        if found['game']:
            best = max(found['game'], key=lambda d: d[0])
            self.cookie_target = (best[1], best[2])
            print(f'[YOLO] печенька ({int(best[1])}, {int(best[2])})')
        if found['upgrade']:
            # зона апгрейдов: самая большая найденная область (правый столбец)
            best = max(found['upgrade'], key=lambda d: d[3] * d[4])
            _, cx, cy, w, h = best
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            # не растягиваем зону на крестик вкладки (верх) и панель задач (низ)
            x1 = max(x1, UPGRADE_X - 120)
            x2 = min(x2, UPGRADE_X + 120)
            self.upgrade_zone = (x1, max(y1, 200), x2, min(y2, UPGRADE_Y_END))
            self.upgrade_zone_found = True
        self.aim_cookie()

    def buy_upgrades(self):
        """Последовательно прокликать апгрейды снизу вверх по прямоугольнику.

        Перед проходом — свежая детекция YOLO: если прямоугольник найден,
        кликаем внутри него; если нет — выводим предупреждение и идём
        по дефолтной зоне (правый столбец).
        """
        old_x, old_y = pyautogui.position()
        print('[Апгрейд]', end=' ', flush=True)

        # свежая детекция прямо перед кликами
        found = self.detect()
        self.handle_detection(found)

        if self.upgrade_zone_found:
            x1, y1, x2, y2 = self.upgrade_zone
            # кнопки апгрейдов прижаты к левому краю зоны (~1500..1560 при 1920),
            # баннеры/пустота справа. Берём фикс. смещение 70px от левого края.
            x_center = min(x1 + 70, UPGRADE_X + 40)
            y_start = max(int(y1) + 130, UPGRADE_Y_START)
            y_end = min(int(y2), UPGRADE_Y_END)  # доходим до самого низа
            origin = f'YOLO ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})'
        else:
            print(' !! прямоугольник не найден, иду по запасной зоне', flush=True)
            x_center = UPGRADE_X
            y_start, y_end = UPGRADE_Y_START, UPGRADE_Y_END
            origin = 'DEfault'

        step = UPGRADE_STEP
        count = 0
        # идём СНИЗУ ВВЕРХ: по одному клику на каждую позицию прямоугольника
        positions = range(y_start, y_end + 1, step)
        for y in reversed(list(positions)):
            jx = x_center + random.randint(-UPGRADE_JITTER, UPGRADE_JITTER)
            jy = y
            pyautogui.moveTo(jx, jy, duration=0.05)  # плавно, без рывков
            win_click()  # нативный клик — быстрее, чем mouseDown/mouseUp
            self.stats['upgrade'] += 1
            count += 1
            time.sleep(UPGRADE_LINE_DELAY)
            print('.', end='', flush=True)

        pyautogui.moveTo(old_x, old_y, duration=0)
        self.last_upgrade_time = time.time()
        print(f' готово. {count} кликов ({origin}), снизу вверх x={x_center:.0f}.')
        self.aim_cookie()  # вернуться на печеньку

    def check_pause(self):
        """Пауза/продолжение по пробелу (срабатывает по нажатию, не удержанию)."""
        pressed = space_pressed()
        if pressed and self._space_up:
            self._space_up = False
            self.paused = not self.paused
            if self.paused:
                print('\n⏸ ПАУЗА (пробел — продолжить)')
            else:
                print('▶ ПРОДОЛЖАЮ')
        elif not pressed:
            self._space_up = True
        return self.paused

    def run(self):
        print('===== КЛИКЕР БОТ =====')
        print(f'Порог: {self.conf}')
        print('Клик: WinAPI (без движения курсора)')
        print(f'Переприцел на печеньку: каждые {COOKIE_CHECK_INTERVAL:.1f}с')
        print(f'Апгрейды: каждые {UPGRADE_INTERVAL:.0f}с (прямоугольник, снизу вверх)')
        print('Пробел: пауза / продолжить')
        print('Стоп: Ctrl+C / мышь в угол экрана')
        print()

        # Первичное наведение: YOLO находит печеньку и ставит на неё курсор
        print('Ищу печеньку...')
        for _ in range(10):
            found = self.detect()
            self.handle_detection(found)
            if self.cookie_target is not None:
                break
            time.sleep(0.25)
        if self.cookie_target is None:
            print('Печенька не найдена. Клики пойдут по текущему положению курсора.')
            self.cookie_pos = pyautogui.position()
        else:
            print(f'Печенька найдена: ({int(self.cookie_target[0])}, {int(self.cookie_target[1])})')

        print('Старт через 1 сек...')
        time.sleep(0.5)
        print('ПОЕХАЛИ!\n')

        next_upgrade = time.time() + UPGRADE_INTERVAL
        next_cookie_aim = time.time() + 1.0

        try:
            while True:
                # Пробел — пауза / продолжить
                if self.check_pause():
                    time.sleep(0.05)
                    continue

                now = time.time()

                # Переприцел на печеньку (YOLO) — если игра подвинула печеньку
                if now >= next_cookie_aim:
                    found = self.detect()
                    next_cookie_aim = now + COOKIE_CHECK_INTERVAL
                    self.handle_detection(found)

                # Апгрейды
                if now >= next_upgrade:
                    self.buy_upgrades()
                    next_upgrade = now + UPGRADE_INTERVAL

                # Клик печеньки (WinAPI — курсор уже наведён)
                win_click()
                self.stats['cookie'] += 1
                time.sleep(CLICK_INTERVAL)

        except KeyboardInterrupt:
            pass
        except pyautogui.FailSafeException:
            pass
        finally:
            total = sum(self.stats.values())
            print(f'\nГотово. cookie={self.stats["cookie"]}, '
                  f'upgrade={self.stats["upgrade"]} '
                  f'(всего {total})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--conf', type=float, default=0.2)
    args = ap.parse_args()
    model_path = find_model_path(args.model)
    print(f'Модель: {model_path}')
    bot = CookieBot(model_path, args.conf)
    bot.run()


if __name__ == '__main__':
    main()
