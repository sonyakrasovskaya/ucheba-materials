#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для создания студенческой версии HW3_v3.ipynb.
Чистит технические принты, приводит комментарии к стилю, сохраняет итоги и сноски.
"""
import json
import re
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "HW3_v3.ipynb"
OUT_PATH = Path(__file__).resolve().parent / "HW3_v3_student_updt.ipynb"

# Паттерны принтов для комментирования (отладочные)
DEBUG_PRINT_PATTERNS = [
    r'print\s*\(\s*["\']?(?:device|Device|cuda|CUDA)',
    r'print\s*\(\s*["\']?generator device',
    r'print\s*\(\s*["\']?discriminator device',
    r'print\s*\(\s*["\']?cuda available',
    r'print\s*\(\s*model\s*\)',
    r'print\s*\(\s*["\']?(?:debug|check|vars_need|found):',
    r'print\s*\(\s*["\']?[12]\)\s+Переменные:',
    r'print\s*\(\s*["\']?Shapes:',
    r'print\s*\(\s*["\']?Mean diff \(abs\)',
    r'print\s*\(\s*["\']?Std diff \(abs\)',
    r'print\s*\(\s*["\']?\\\\n=== [0-9]+\)',
    r'print\s*\(\s*["\']?Generator определён',
    r'print\s*\(\s*["\']?Discriminator определён',
    r'print\s*\(\s*["\']?Fitter определён',
    r'print\s*\(\s*["\']?generate определён',
    r'print\s*\(\s*["\']?qt определён',
    r'print\s*\(\s*["\']?DiffusionGenerator определён',
    r'print\s*\(\s*["\']?forward сигнатура',
    r'print\s*\(\s*["\']?generate_with_diffusion определён',
    r'print\s*\(\s*["\']?Есть t_batch',
    r'print\s*\(\s*["\']?Модель вызывается',
    r'print\s*\(\s*["\']?num_inference_steps',
    r'print\s*\(\s*["\']?safe_save: вызов',
    r'print\s*\(\s*["\']?safe_load_model:',
    r'print\s*\(\s*f["\']?safe_load_model:',
    r'print\s*\(\s*["\']?vae_fitter\.decoder',
    r'print\s*\(\s*["\']?generate вернул',
    r'print\s*\(\s*["\']?проверка типа',
    r'print\s*\(\s*["\']?проверка формы',
    r'print\s*\(\s*["\']?cVAE generate',
    r'print\s*\(\s*["\']?inspect\.getsource',
    r'print\s*\(\s*["\']?Проверь вручную',
    r'print\s*\(\s*["\']?Шаг 2 закрыт',
    r'print\s*\(\s*["\'][\\\\]nШаг 2 закрыт',  # " затем \n в строке
    r'print\s*\(\s*["\']?\s*Шаг 2 закрыт',
    r'print\s*\(\s*["\']?Блок 3 закрыт',
    r'print\s*\(\s*["\'][\\\\]nБлок 3 закрыт',
    r'print\s*\(\s*["\']?\s*Блок 3 закрыт',
    r'print\s*\(\s*["\']?\\\\nШаг 2 закрыт',
    r'print\s*\(\s*["\']?\\\\nБлок 3 закрыт',
    r'print\s*\(\s*["\']?3\) Размеры для ROC',
    r'print\s*\(\s*["\']?4\) Регуляризация',
    r'print\s*\(\s*["\']?5\) NaN',
    r'print\s*\(\s*["\']?6\) Размеры XX',
    r'print\s*\(\s*["\']?6\) Ошибка',
    r'print\s*\(\s*["\']?6\) XX_train',
    r'print\s*\(\s*["\']?1\) generate есть',
    r'print\s*\(\s*["\']?2\) Переменные',
    r'print\s*\(\s*["\']?3\) Формы',
    r'print\s*\(\s*["\']?3\) [A-Za-z_]+: ожидается',
    r'print\s*\(\s*["\']?4\) В ',
    r'print\s*\(\s*["\']?4\) NaN',
    r'print\s*\(\s*["\']?5\) ROC-AUC рассчитан',
    r'print\s*\(\s*["\']?\\\\nШаг 2 закрыт',
    r'print\s*\(\s*["\']?\\\\nБлок 3 закрыт',
    r'print\s*\(\s*["\']?\\\\n=== 2\)',
    r'print\s*\(\s*["\']?\\\\n=== 3\)',
    r'print\s*\(\s*["\']?\\\\n=== 4\)',
    r'print\s*\(\s*["\']?\\\\n=== 5\)',
    r'print\s*\(\s*["\']?\\\\n=== Итог',
    r'print\s*\(\s*["\']?Если WGAN и cVAE',
    r'print\s*\(\s*["\']?Если diffusion не принимает',
    r'print\s*\(\s*["\']?форма:',
    r'print\s*\(\s*["\']?qt\.inverse_transform есть',
    r'print\s*\(\s*["\']?qt не найден',
    r'print\s*\(\s*["\']?не удалось прочитать сигнатуру',
    r'print\s*\(\s*["\']?inspect\.getsource не сработал',
    r'print\s*\(\s*["\']?- есть ли t_batch',
    r'print\s*\(\s*["\']?- вызывается ли model',
    r'print\s*\(\s*["\']?- стоит ли scheduler',
    r'print\s*\(\s*["\']?Diffusion forward ok',
    r'print\s*\(\s*["\']?model \(diffusion\) ещё не создан',
    r'переменная не найдена',
    r'есть NaN или inf',
    r'ожидается 10 признаков',
    r'ожидается \(n, 10\)',
    r'В .* есть NaN или inf',
    r'XX_train/XX_test или yy_',
]
# многострочный print: строка "print(" одна на строке
PRINT_OPEN = re.compile(r'^\s*print\s*\(\s*$')
SHAPE_MINMAX_PATTERN = re.compile(
    r'print\s*\(\s*f?\s*["\']?\s*\{\s*["\']?\s*name\s*["\']?\s*\}\s*:\s*shape='
)

# Оставляем без изменений
KEEP_PRINT_PATTERNS = [
    r'ROC AUC',
    r'roc_auc|roc auc',
    r'sanity|Sanity',
    r'saved:',
    r'Пропуск|пропуск|пропущен|Тяжёлый блок пропущен',
    r'Лучшие 20 строк',
    r'Лучший набор для',
    r'Время:.*сек',
    r'Это быстрый подбор',
]

def should_remove_print(line: str) -> bool:
    stripped = line.strip()
    if 'print(' not in stripped and 'print (' not in stripped:
        return False
    for pat in KEEP_PRINT_PATTERNS:
        if re.search(pat, line, re.IGNORECASE):
            return False
    for pat in DEBUG_PRINT_PATTERNS:
        if re.search(pat, line):
            return True
    # только в строках с print — дополнительные фразы
    if 'print(' in line or 'print (' in line:
        for phrase in ['переменная не найдена', 'есть NaN или inf', 'ожидается 10 признаков',
                       'ожидается (n, 10)', 'есть NaN или inf', 'XX_train/XX_test или yy_']:
            if phrase in line:
                return True
    if SHAPE_MINMAX_PATTERN.search(line):
        return True
    if 'shape=' in line and 'min=' in line and 'max=' in line and 'print' in line:
        return True
    return False

def comment_out_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith('#'):
        return line
    indent = line[: len(line) - len(stripped)]
    return indent + '# ' + stripped

def fix_comment_style(line: str) -> str:
    if '#' not in line:
        return line
    parts = line.split('#', 1)
    if len(parts) != 2:
        return line
    before, after = parts[0], '#' + parts[1]
    after = after.replace('—', '-')
    after = re.sub(r'Источник\s*:', 'источник:', after, flags=re.IGNORECASE)
    m = re.match(r'^(\#\s*)([А-ЯA-Z])(.*)$', after)
    if m:
        prefix, first, rest = m.group(1), m.group(2), m.group(3)
        if first in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ':
            first = first.lower()
        elif first in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and rest.startswith(' ') and len(rest) > 2 and rest[1:2].islower():
            first = first.lower()
        after = prefix + first + rest
    for pat, repl in [
        (r'\bскейл\b', 'масштабирование'),
        (r'\bсемплинг\b', 'порождение'),
        (r'\bинференс\b', 'прогон'),
        (r'\bдевайс\b', 'устройство'),
        (r'\bфорвард\b', 'прямой проход'),
    ]:
        after = re.sub(pat, repl, after, flags=re.IGNORECASE)
    return before + after

def process_cell(cell: dict, index: int) -> None:
    if cell.get('cell_type') == 'markdown':
        src = cell.get('source', [])
        if isinstance(src, str):
            src = [src]
        full = ''.join(src)
        # заголовки моделей: один # -> ## для единой структуры
        full = full.replace('# Conditional WGAN', '## Conditional WGAN')
        full = full.replace('# Условные вариационные автокодировщики', '## Условные вариационные автокодировщики')
        full = full.replace('# Диффузионные модели', '## Диффузионные модели')
        full = full.replace('# Нормализационные потоки', '## Нормализационные потоки')
        # единая структура секций: заменяем весь текст задания на короткий заголовок
        replacements_md = [
            ('## Задание 1 (0.5 балла)', '### реализация модели\n\nпредобработка QuantileTransformer.'),
            ('## Задание 2 (0.25 балла)', '### архитектура\n\nреализация генератора и дискриминатора (условные по y).'),
            ('## Задание 3 (0.25 балла)', '### архитектура\n\nреализация дискриминатора (условный по y).'),
            ('## Задание 4 (1 балл)', '### обучение\n\nцикл обучения WGAN-GP (Fitter).'),
            ('## Обучение\nОбучим модель на данных.', '### обучение\n\nзапуск обучения на данных.'),
            ('## Задание 5 (0.5 балла)', '### генерация и оценка\n\nгенерация сэмплов по y и расчёт ROC-AUC.'),
            ('## Задание 6 (0.5 балла)', '### архитектура\n\nэнкодер и декодер (условный cVAE).'),
            ('## Задание 7 (0.5 балла)', '### архитектура\n\nдекодер (восстановление по z и y).'),
            ('## Задание 8 (0.5 балл)', '### обучение\n\nVAEFitter и цикл обучения.'),
            ('## Задание 9 (0.5 балл)', '### генерация и оценка\n\nпорождение сэмплов и ROC-AUC.'),
            ('## Задание 10 (0.5 балла)', '### архитектура\n\nпрямой процесс зашумления (corrupt) и модель.'),
            ('## Задание 11 (0.5 балла)', '### архитектура\n\nмодель DiffusionGenerator (x_noisy, y, t).'),
            ('## Задание 12 (0.5 балла)', '### генерация и оценка\n\nфункция генерации (обратный процесс).'),
            ('## Задание 13 (1 балла)', '### обучение\n\nцикл обучения диффузии.'),
            ('## Задание 14 (0.5 балла)', '### генерация и оценка\n\nгенерация и ROC-AUC.'),
            ('## Задание 15 (1 балл)', '### архитектура\n\nнормализующий поток (слои и априор).'),
            ('## Задание 16 (1 балл)', '### обучение\n\nфункция train_nf и обучение потока.'),
            ('## Задание 17 (0.5 балла)', '### генерация и оценка\n\nгенерация и ROC-AUC (в ячейке выше).'),
        ]
        for old, new in replacements_md:
            if old in full:
                full = new
                break
        full = full.replace('—', '-')
        full = re.sub(r'Источник\s*:', 'источник:', full, flags=re.IGNORECASE)
        new_src = [line + '\n' for line in full.split('\n')]
        if new_src and not full.endswith('\n'):
            new_src[-1] = new_src[-1].rstrip('\n') or new_src[-1]
        cell['source'] = new_src
        return

    if cell.get('cell_type') != 'code':
        return

    src = cell.get('source', [])
    if isinstance(src, str):
        src = [src]
    new_src = []
    i = 0
    while i < len(src):
        line = src[i]
        # замена YOUR CODE IS HERE на ### реализация
        if '### YOUR CODE IS HERE' in line or '### YOUR CODE IS HERE ######' in line:
            line = re.sub(r'#+\s*YOUR CODE IS HERE\s*#*', '### реализация', line)
        if '### THE END OF YOUR CODE ###' in line or '### THE END OF YOUR CODE' in line:
            line = re.sub(r'#+\s*THE END OF YOUR CODE\s*#*', '', line).rstrip()
            if not line.strip():
                i += 1
                continue
        # убрать вывод DEVICE (ячейка только "DEVICE")
        if line.strip() == 'DEVICE' and len(src) == 1:
            line = '# DEVICE  # устройство задано выше\n'
        line = fix_comment_style(line)
        # многострочный print( ... ) — комментируем весь блок
        if PRINT_OPEN.match(line.strip()):
            new_src.append(comment_out_line(line))
            i += 1
            while i < len(src):
                next_line = src[i]
                st = next_line.strip()
                if st.startswith(')'):
                    new_src.append(comment_out_line(next_line))
                    i += 1
                    break
                if st.endswith(',') or ('shape' in next_line and 'min=' in next_line) or ('X_fake_test_orig' in next_line and 'shape' in next_line):
                    new_src.append(comment_out_line(next_line))
                    i += 1
                else:
                    break
            continue
        if should_remove_print(line):
            line = comment_out_line(line)
        new_src.append(line)
        i += 1
    cell['source'] = new_src

def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb.get('cells', [])):
        process_cell(cell, idx)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Saved: {OUT_PATH}")

if __name__ == '__main__':
    main()
