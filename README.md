# Dino Runner Bot

Q-learning агент, который играет в [T-Rex Runner](https://wayou.github.io/t-rex-runner/)
через Playwright (Firefox). Обучение идёт через ε-жадную стратегию по таблице
Q-значений, состояние игры читается напрямую из JS-объекта `window.Runner`.

## Что внутри

- [`dino.py`](dino.py) — основной бот: `DinoAgent` (Q-learning), `GameSession`
  (браузер/игра), `TrainingLog` (CSV-журнал).
- [`dino_backup.py`](dino_backup.py) — исходная (более простая) версия скрипта.

## Состояние

- **Реакция**: прыжок / присед / нет препятствия.
- **Дистанция**: 4 зоны (далеко / средне / близко / критично).
- Итого 12 состояний × 3 действия (`NONE`, `JUMP`, `DUCK`).

## Запуск

```bash
# обучение (50 эпизодов по умолчанию)
python dino.py

# короткое обучение
python dino.py --episodes 10

# только играть сохранённой моделью (без обучения)
python dino.py --eval

# браузер без окна
python dino.py --headless
```

## Требования

- Python 3.10+
- `playwright` с установленным Firefox: `playwright install firefox`
- `numpy`

## Файлы

- `q_table.npy` — сохранённая Q-таблица (создаётся автоматически).
- `training_log.csv` — журнал результатов по эпизодам.
