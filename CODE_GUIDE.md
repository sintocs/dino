# Полный разбор кода Dino NEAT — учебник

> Для студента 1 курса. Читай сверху вниз, не перескакивая: каждый следующий блок
> опирается на предыдущий.

---

## 0. Главная картина — запомни её первой

В проекте **три этажа**, и они общаются только через два «моста»:

```
┌──────────────────────────────────────────────────────────────┐
│ ЭТАЖ 1 — Python: эволюция (NEAT)                             │
│ поколение → 200 геномов → у каждого fitness → отбор → дети   │
└───────────────────────────┬──────────────────────────────────┘
                            │  мост №1: network_to_js()
                            │  (Python-сеть → обычный JSON)
┌───────────────────────────▼──────────────────────────────────┐
│ ЭТАЖ 2 — Playwright: 15 браузеров Chromium                   │
│ каждый браузер = отдельная страница с игрой                  │
└───────────────────────────┬──────────────────────────────────┘
                            │  мост №2: page.evaluate()
                            │  (Python просит JS выполнить функцию)
┌───────────────────────────▼──────────────────────────────────┐
│ ЭТАЖ 3 — JavaScript внутри страницы (AGENT_JS)               │
│ игра + нейросеть + подсчёт прыжков → возвращает результат    │
└──────────────────────────────────────────────────────────────┘
```

**Ключевая мысль, без которой ничего не поймёшь:**

> Python **не играет** в игру. Python — это «тренер». Он говорит браузеру:
> «вот чертёж сети, прогони эпизод и скажи счёт». Браузер играет **сам**,
> внутри себя, и возвращает готовый результат.

Почему так? Потому что игра уже написана на JavaScript. Гонять каждый кадр
(60 раз в секунду) из Python в браузер и обратно — это огромная задержка.
Поэтому сеть **копируется внутрь страницы** и работает там же, где игра.

### Как запомнить три этажа

| Этаж | Кто | Что делает | Аналогия |
|------|-----|-----------|----------|
| 1 | Python | решает, кто размножается | тренер / селекционер |
| 2 | Playwright | открывает 15 окон | стадион с дорожками |
| 3 | JS | играет и считает очки | спортсмен |

---

## 1. Что такое NEAT за 30 секунд

**NEAT** (NeuroEvolution of Augmenting Topologies) — это эволюция **не чисел,
а чертежей сети**.

Обычная нейросеть: структура фиксирована, учим только веса.
NEAT: структура **тоже мутирует** — может добавить нейрон, убрать связь,
сменить функцию активации.

Цикл, как в природе:

```
1. Есть популяция геномов (200 штук) — «особи»
2. Каждый играет → получает fitness (оценку)
3. Лучшие → дают потомство (с мутациями)
4. Плохие → умирают
5. Повторить
```

**Геном** — это словарь:
- `nodes` — нейроны (с bias, функцией активации)
- `connections` — связи (от кого, к кому, вес, включена ли)
- `fitness` — оценка (появляется после игры)

---

## 2. `dino.py` — разбор по блокам

### Блок 1. Docstring и импорты (строки 1–53)

```python
"""Dino Runner: обучение нейросети NEAT в 15 браузерах Chromium одновременно."""
```

Это **документация прямо в коде**. Тройные кавычки в начале файла = docstring.
Он объясняет, что изменено по сравнению со старой версией (там был Q-learning).

```python
from __future__ import annotations   # можно писать list[int] вместо List[int]
import argparse                      # разбор аргументов командной строки
import asyncio                       # асинхронность (несколько браузеров сразу)
import csv                           # журнал обучения
import json                          # сохранение сети
import os                            # файлы и папки
import pickle                        # сохранение генома (Python-объект в файл)
import random                        # случайные сиды
import time                          # замер времени

from dataclasses import dataclass    # быстрый класс для данных
from typing import Any, Optional

import neat                          # библиотека NEAT
from neat.graphs import feed_forward_layers   # раскладка сети по слоям
from playwright.async_api import (...)        # браузер
```

**🧠 Запомни:** `pickle` — это «заморозить Python-объект в файл», `json` —
«заморозить в текст, понятный всем».

```python
GAME_URL = "https://wayou.github.io/t-rex-runner/"
NEAT_CONFIG = "neat_config.ini"
```

Две константы: откуда грузить игру и какой конфиг читать.

---

### Блок 2. Геометрия мира (строки 55–66)

```python
CANVAS_W = 600
CANVAS_H = 150
GROUND_Y = 93          # верхняя граница дино на земле
DINO_H = 47
DINO_H_DUCK = 25
DINO_W = 44
MAX_SPEED = 13
START_SPEED = 6
SCORE_COEF = 0.025
```

Это **размеры из исходника игры**. Зачем они в Python, если игра в JS?
Потому что некоторые нужны для `agent_cfg` и для нормализации признаков.

**🧠 Запомни:** `y` в canvas растёт **вниз**. Поэтому «высота над землёй» =
`groundYPos - yPos`.

---

### Блок 3. Признаки сети — 19 входов (строки 68–92)

```python
FEATURE_NAMES = [
    "dx1_norm",         # расстояние до 1-го препятствия
    "dx2_norm",         # расстояние до 2-го
    "has_second",       # есть ли второе
    "is_ptero",         # птица?
    "ptero_y_norm",     # высота птицы
    "is_cactus_large",
    "is_cactus_small",
    "obs_height_norm",
    "obs_width_norm",
    "obs_top_norm",
    "obs_bottom_norm",
    "gap_after_norm",   # зазор справа от препятствия
    "dino_airborne",    # высота дино над землёй
    "jump_velocity",
    "jumping",
    "ducking",
    "speed_drop",
    "speed_norm",
    "blocks_ducking",   # помешает ли даже приседу
]
NUM_INPUTS = len(FEATURE_NAMES)   # = 19
```

**Зачем список в Python, если признаки собираются в JS?**
Это **контракт**. Порядок обязан совпадать с `buildInputs()` в JS. Если
перепутаешь — сеть будет «видеть» высоту как расстояние и никогда не научится.

**🧠 Запомни:** 19 входов, 2 выхода (`прыжок`, `присед`). Это как глаза
(19 сигналов) и руки (2 действия).

---

### Блок 4. `AGENT_JS` — программа для страницы (строки 94–578)

Это огромная строка Python (`r""" ... """`), внутри которой **JavaScript**.
Она один раз вставляется в страницу через `add_init_script()` и живёт там.

Разберём её изнутри.

#### 4.1 Защита от повторной вставки

```javascript
if (window.__dino) { return; }
```

Если скрипт уже стоит — второй раз не ставим. Иначе переменные затрутся.

#### 4.2 `DEFAULTS` и `FRAME_MS`

```javascript
const DEFAULTS = {
  canvasWidth: 600, seconds: 30, maxFrames: 4000,
  clearMs: 900, seed: 1, render: false
};
const FRAME_MS = 1000 / 60;   // ≈ 16.67 мс — один кадр при 60 FPS
```

`clearMs` — пауза в начале, пока пустая дорога.

#### 4.3 `ACT` — функции активации

```javascript
const ACT = {
  sigmoid: z => 1.0 / (1.0 + Math.exp(-clamp(...))),
  tanh:    z => Math.tanh(clamp(...)),
  relu:    z => (z > 0 ? z : 0),
  ...
};
```

**Зачем все, если в конфиге только 3?**
NEAT может мутировать функцию активации. Python присылает её **именем**
(`"tanh"`, `"relu"`). JS должен уметь выполнить любое имя, иначе упадёт.

**🧠 Запомни:** это «словарь команд». Python говорит `act: "relu"` →
JS берёт `ACT["relu"]` и вызывает.

#### 4.4 `AGG` — как складывать входы

```javascript
const AGG = {
  sum:     xs => xs.reduce((a, b) => a + b, 0),
  mean:    xs => ...,
  max:     xs => ...,
  min:     xs => ...,
  ...
};
```

Нейрон получает несколько входов. Их можно сложить, усреднить, взять максимум.
Тоже мутирует через конфиг.

#### 4.5 Состояние модуля

```javascript
let NET = null;      // сеть (приходит из Python)
let CFG = {...};     // настройки эпизода
let state = 'idle';  // idle | running | done
let result = null;   // результат последнего эпизода
let run = null;      // счётчики текущего прогона
let seed = 1;
```

**🧠 Запомни:** `state` — это «светофор»: `idle` (ждём), `running` (играем),
`done` (готово).

#### 4.6 `installSeededRandom()` — детерминированный случай (строки 163–172)

```javascript
function installSeededRandom() {
  let s = seed >>> 0;
  Math.random = function () {          // подменяем встроенный Math.random
    s = (s + 0x6D2B79F5) >>> 0;
    ...
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

Это **mulberry32** — маленький ГПСЧ. Мы **подменяем** `Math.random` во всей
странице.

**Зачем?** Игра строит препятствия через `Math.random()`. Если он настоящий
случайный — каждый геном получает **разную** трассу, и сравнение нечестное
(одному повезло, другому нет). С фиксированным сидом все играют **одинаково**.

**🧠 Запомни:** сид = «зерно». Одно зерно → одна и та же последовательность
случайных чисел → одна и та же трасса.

#### 4.7 `activate(inputs)` — прямой проход сети (строки 174–192)

```javascript
function activate(inputs) {
  const values = {};
  for (let i = 0; i < NET.inputs.length; i++)
    values[NET.inputs[i]] = inputs[i] || 0;

  for (let n = 0; n < NET.evals.length; n++) {
    const node = NET.evals[n];
    const xs = [];
    for (let k = 0; k < node.links.length; k++) {
      const inode = node.links[k][0];
      const w = node.links[k][1];
      xs.push((values[inode] || 0) * w);       // вход × вес
    }
    const agg = AGG[node.agg] || AGG.sum;
    const act = ACT[node.act] || ACT.tanh;
    values[node.node] = act(node.bias + node.response * agg(xs));
  }
  return NET.outputs.map(k => values[k]);
}
```

Разбор по шагам:

1. Кладём 19 входов в словарь `values` под их номерами.
2. Идём по `NET.evals` — это **уже отсортированный** список нейронов
   (Python разложил по слоям). Поэтому к моменту вычисления нейрона все его
   входы уже посчитаны.
3. Для нейрона: каждый вход × вес → список `xs`.
4. `agg(xs)` — сложить/усреднить.
5. `act(bias + response * сумма)` — функция активации.
6. Возвращаем значения выходов.

**🧠 Запомни формулу нейрона:**

```
выход = активация( bias + response × агрегация(Σ вход×вес) )
```

Это сердце всей нейросети. Всё остальное — обвязка.

#### 4.8 `buildInputs(runner)` — глаза (строки 194–241)

```javascript
const f = new Array(19).fill(0);
const dnorm = x => clamp(x / 160, -1, 3);
```

Нормировка на **160 px**, а не на 600. Почему? Потому что динозавр реагирует
на ближнюю зону. Если делить на 600, то 400 px и 120 px дадут 0.67 и 0.20 —
сеть почти не различает. А если на 160 — получим 2.5 и 0.75, разница огромная.

Дальше по индексам:

| Индекс | Признак | Как считается |
|--------|---------|---------------|
| 0 | расстояние до 1-го | `dnorm(obs.xPos - dino.xPos)` |
| 1 | до 2-го | или ширина канваса |
| 2 | есть второе | 1/0 |
| 3 | птица? | 1/0 |
| 4 | высота птицы | `oy / 150` |
| 5,6 | тип кактуса | 1/0 |
| 7,8 | высота/ширина | `/50` |
| 9,10 | верх/низ препятствия | `/150` |
| 11 | зазор справа | `dnorm(dx1 - width)` |
| 18 | блокирует присед | `oy+oh > 118 && oy < 93` |

Дино-часть:

```javascript
f[12] = clamp(airborne / 50, 0, 1);   // высота над землёй
f[13] = jumpVelocity / 15;            // скорость прыжка
f[14] = dino.jumping ? 1 : 0;
f[15] = dino.ducking ? 1 : 0;
f[16] = dino.speedDrop ? 1 : 0;
f[17] = (speed - 6) / 7;              // 6..13 → 0..1
```

**🧠 Запомни:** все числа приводятся к диапазону примерно **−1…1**. Нейросети
не любят большие числа.

#### 4.9 `apply(runner, jump, duck)` — руки (строки 243–269)

```javascript
const doJump = jumpValue > 0.5;
const doDuck = !doJump && duckValue > 0.5;
```

Выход сети — число. **Порог 0.5** превращает его в действие.

Логика:
- прыжок → если приседал, встаём; если не в прыжке — прыгаем, `jumps++`;
- присед → если в прыжке, `setSpeedDrop()` (быстрое падение), иначе приседаем;
- ничего → если приседал, встаём.

```javascript
} catch (err) { /* игра в состоянии рестарта — пропускаем кадр */ }
```

Защита: игра может быть не готова, не падаем.

#### 4.10 `trackJump(runner)` — «полезный» прыжок (строки 271–293)

```javascript
if (!dino.jumping) {
  run.jumpPending = false;
  if (!run.jumpSafe) { run.useless += 1; }   // приземлился, а препятствие не прошло
  return;
}
// пока в воздухе — проверяем пересечение по X
if (o.xPos < dinoRight && (o.xPos + o.width) > dinoLeft) {
  run.jumpSafe = true;
}
```

**Зачем?** Без штрафа NEAT находит «тупой» оптимум: **прыгать без остановки**.
Это даёт 60–90 очков и всё, дальше эволюция не идёт. Штраф убирает
прыжки-в-пустоту.

**🧠 Запомни:** `jumpPending` = «прыгнул, ждём приземления»,
`jumpSafe` = «под нами реально прошло препятствие».

#### 4.11 `resetGame(runner, clock, noRender)` (строки 310–358)

```javascript
runner.stop();                  // убить родной цикл игры
runner.crashed = false;
runner.playing = true;
runner.distanceRan = 0;
runner.setSpeed(runner.config.SPEED);
...
horizon.obstacles = [];          // ВАЖНО: полный сброс
horizon.obstacleHistory = [];
horizon.clouds = [];
...
if (noRender) {                  // отключаем рисование — быстрее
  runner.clearCanvas = function () {};
  ctx.drawImage = function () {};
}
runner.restart();
```

**Тонкость, за которую дают «плюс»:** родной `horizon.reset()` **не чистит**
`clouds`. Облака тоже дёргают `Math.random()` — и сдвигают поток случайных
чисел. Трасса получалась разной от прогона к прогону. Поэтому чистим вручную.

**🧠 Запомни:** «сброс должен быть полным, иначе сид ломается».

#### 4.12 `simulate()` — сердце обучения (строки 380–437)

```javascript
function simulate() {
  installSeededRandom();

  const realNow = performance.now.bind(performance);
  let clock = 1000;
  performance.now = () => clock;       // ← ЗАМОРАЖИВАЕМ ЧАСЫ
  ...
  resetGame(runner, clock, !CFG.render);

  const maxByTime   = Math.ceil((CFG.seconds || 30) * 60);
  const maxByFrames = CFG.maxFrames > 0 ? CFG.maxFrames : Infinity;
  const frameLimit  = Math.min(maxByTime, maxByFrames);

  for (let i = 0; i < frameLimit; i++) {
    if (runner.crashed) { status = 'DEAD'; break; }
    const outputs = activate(buildInputs(runner));   // 1. думаем
    apply(runner, outputs[0], outputs[1]);           // 2. действуем
    clock += FRAME_MS;                               // 3. двигаем время
    runner.update();                                 // 4. двигаем мир
    trackJump(runner);                               // 5. оцениваем прыжок
    run.frames = i + 1;
  }
  performance.now = realNow;    // ← возвращаем настоящие часы
  ...
}
```

Это **главный трюк проекта**. Обычно игра идёт в реальном времени: 1 кадр =
16 мс настоящего времени. Мы **подменяем `performance.now`** так, чтобы игра
думала, что время идёт ровно по 16.67 мс за шаг, а сами крутим цикл
**максимально быстро**.

Результат: эпизод на 60 секунд считается за **десятки миллисекунд** — в ~40 раз
быстрее. И полностью воспроизводимо.

**🧠 Запомни порядок шага:** `думаем → действуем → время → мир → оценка`.

#### 4.13 `startLive()` — то же, но с анимацией (строки 456–545)

```javascript
const nowReal = realNow();
acc += nowReal - last;
let steps = Math.floor(acc / FRAME_MS);
if (steps > 4) { steps = 4; acc = 0; }   // защита от «спирали смерти»
```

Физика **та же** (шаг 1/60), но темп задаёт `requestAnimationFrame` — чтобы
было видно игру. Ограничение 4 шага: если окно подвисло, игра не будет
догонять 10 секунд за один кадр.

Используется в `--eval`.

#### 4.14 Экспорт (строки 569–576)

```javascript
window.__dino = { init, start, poll, abort, info, simulate };
```

`window.__dino` — это **пульт управления**, который Python дёргает через
`page.evaluate("() => window.__dino.simulate()")`.

---

### Блок 5. `network_to_js()` — мост №1 (строки 584–617)

```python
def network_to_js(genome, config) -> dict[str, Any]:
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers, required = feed_forward_layers(
        genome_config.input_keys, genome_config.output_keys, connections
    )
```

Что происходит:

1. Берём **только включённые** связи (`if cg.enabled`) — выключенные не работают.
2. `feed_forward_layers()` — раскладывает сеть по слоям **в правильном порядке**.
   Возвращает `layers` (слои) и `required` (какие нейроны реально нужны).
3. Для каждого нейрона собираем входящие связи: `[[откуда, вес], ...]`.
4. Отдаём словарь:

```python
{
  "inputs":  [...],   # номера входов
  "outputs": [...],   # номера выходов
  "evals":   [        # нейроны в порядке вычисления
      {"node": 5, "act": "tanh", "agg": "sum",
       "bias": 0.3, "response": 1.0, "links": [[1, -0.7], [3, 2.1]]},
      ...
  ]
}
```

**🧠 Запомни:** это «перевод чертежа с языка Python на язык JSON», чтобы
браузер мог сеть выполнить.

---

### Блок 6. `EpisodeResult` — результат (строки 623–633)

```python
@dataclass
class EpisodeResult:
    fitness: float
    score: int
    frames: int
    seconds: float
    status: str
    jumps: int = 0
    useless: int = 0
    ducks: int = 0
    error: str = ""
```

`@dataclass` — декоратор, который сам пишет `__init__`, `__repr__`. Экономим
20 строк.

**Статусы:** `DEAD` (разбился), `TIME` (дожил до лимита времени),
`FRAMES` (дожил до лимита кадров), `ERROR`, `NO_RUNNER`, `PY_TIMEOUT`, `ABORT`.

---

### Блок 7. `make_result()` — считаем fitness (строки 636–674)

```python
USELESS_JUMP_PENALTY = 0.6
SURVIVAL_BONUS = 250.0
```

```python
useful_share = 1.0 if jumps == 0 else max(0.0, 1.0 - useless / max(1, jumps))
fitness  = float(score) * (0.4 + 0.6 * useful_share) + 0.02 * frames
fitness -= USELESS_JUMP_PENALTY * useless
if status in ("TIME", "FRAMES"):
    fitness += SURVIVAL_BONUS + 0.25 * score
elif status in ("NO_RUNNER", "STALL", "ERROR", "ABORT", "PY_TIMEOUT"):
    fitness = -100.0
```

Разберём по-человечески:

| Компонент | Смысл |
|-----------|-------|
| `score × (0.4 + 0.6 × useful_share)` | основной вклад — очки, но за «мусорные» прыжки режем |
| `+ 0.02 × frames` | микро-бонус за каждый кадр (плавность) |
| `− 0.6 × useless` | штраф за прыжки в пустоту |
| `+ 250 + 0.25 × score` | дожил до лимита — бонус |
| `= −100` | техническая ошибка — сильно наказываем |

`useful_share` — доля полезных прыжков. Все прыжки полезны → 1.0 → множитель
1.0. Половина мусорных → 0.7 → множитель 0.82.

**🧠 Запомни:** fitness ≠ score. Score — это очки в игре. Fitness — это
**то, по чему учится эволюция** (score + бонусы − штрафы).

> Мелочь для внимательных: в README написано `0.3 + 0.7`, а в коде
> `0.4 + 0.6`. Документация отстала от кода — так бывает в реальных проектах.

---

### Блок 8. `BrowserWorker` — один браузер (строки 680–884)

#### 8.1 `__init__`

```python
self.index = index          # номер воркера (0..14)
self.browser = None         # объект браузера
self.context = None         # контекст (как «профиль»)
self.page = None            # вкладка
self.net = None             # текущая сеть в JSON
self.route_seed = ...
self.agent_cfg = {
    "canvasWidth": CANVAS_W,
    "seconds": args.episode_seconds,
    "maxFrames": args.max_frames,
    "clearMs": args.clear_ms,
    "seed": self.route_seed if ... else (1 + index),
}
```

**🧠 Запомни:** воркер = браузер + контекст + страница + сеть. Всё своё,
не пересекается с другими.

#### 8.2 `start()`

```python
launch_args = [
    "--mute-audio",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]
self.browser = await self.playwright.chromium.launch(
    headless=self.args.headless, args=launch_args)
self.context = await self.browser.new_context(
    viewport={"width": 1000, "height": 500}, device_scale_factor=1)
self.page = await self.context.new_page()
await self.page.add_init_script(script=AGENT_JS)   # ← вставляем наш JS
await self._open_game()
```

Три флага отключения throttling — **критично**. Chromium «засыпает»
фоновые вкладки: таймеры замедляются в 10 раз. Без этих флагов 15 браузеров
будут тормозить.

`add_init_script` — скрипт выполняется **при каждой загрузке** страницы,
до кода игры.

#### 8.3 `_open_game()`

```python
await self.page.goto(GAME_URL, wait_until="load", timeout=45_000)
await self.page.wait_for_function(
    "() => !!(window.Runner && window.Runner.instance_ && window.Runner.imageSprite)",
    timeout=45_000)
await self.page.wait_for_function(
    "() => !!(window.__dino && window.__dino.info().ready)", timeout=15_000)
```

Двойное ожидание:
1. игра загрузилась (`Runner.instance_`, `imageSprite`);
2. наш агент готов (`__dino.info().ready`).

**🧠 Запомни:** `wait_for_function` — «жди, пока JS-условие не станет true».

#### 8.4 `_recover()` — починить страницу

```python
try:
    await self._open_game()             # сначала просто перезагрузить
    if self.net: await page.evaluate("init", ...)
except Exception:
    await self.browser.close()          # не помогло — перезапустить браузер
    await self.start()
```

Два уровня: **мягкий** (reload) и **жёсткий** (новый браузер).

#### 8.5 `run_genome()` — главный метод воркера

```python
self.net = network_to_js(genome, self.config)
if self.args.live:
    return await self._run_genome_live()

hard_timeout = max(30.0, self.args.episode_seconds * 1.5 + 15.0)
total_timeout = hard_timeout + 20.0

for attempt in range(1, 4):          # 3 попытки
    try:
        if attempt > 1: await self._recover()
        await self.page.evaluate("(p) => window.__dino.init(p)",
                                 {"net": self.net, "cfg": self.agent_cfg})
        snapshot = await asyncio.wait_for(
            self.page.evaluate("() => window.__dino.simulate()"),
            timeout=total_timeout)
        return make_result(snapshot["result"])
    except asyncio.TimeoutError:
        await self._force_restart()      # браузер завис → перезапуск
    except Exception as error:
        ...
return EpisodeResult(-100.0, 0, 0, 0.0, "ERROR", "retries exhausted")
```

Три уровня защиты:
1. `asyncio.wait_for(..., timeout=)` — не ждём вечно;
2. `attempt` 1..3 — повтор с восстановлением;
3. после всех попыток — `-100` (не роняем обучение).

**🧠 Запомни:** «воркер никогда не падает, он возвращает плохой результат».

#### 8.6 `_force_restart()`

```python
await self.browser.close()
await self.start()
if self.net: await page.evaluate("init", ...)
```

Полный перезапуск Chromium, когда страница не отвечает вообще.

#### 8.7 `close()`

```python
try: await self.browser.close()
except Exception: pass
```

Всегда закрываем браузер, иначе процессы останутся висеть в памяти.

---

### Блок 9. `AsyncEvaluator` — диспетчер (строки 890–958)

```python
class AsyncEvaluator:
    def __init__(...):
        self.workers = []
        self.results = {}
```

#### 9.1 Вход/выход через `async with`

```python
async def __aenter__(self):
    self.workers = [BrowserWorker(i, ...) for i in range(self.args.browsers)]
    await asyncio.gather(*(w.start() for w in self.workers))
    return self

async def __aexit__(self, *exc_info):
    await asyncio.gather(*(w.close() for w in self.workers))
```

`async with` гарантирует: даже при ошибке все браузеры закроются.

**🧠 Запомни:** `asyncio.gather` — «запусти всё одновременно и дождись всех».

#### 9.2 `set_route_seed()`

```python
for worker in self.workers:
    worker.agent_cfg["seed"] = seed + worker.index
```

У всех воркеров **разные, но близкие** сиды. Почему? Чтобы сеть видела
разные трассы и **обобщала**, а не заучивала одну карту.

#### 9.3 `evaluate()` — раздача работы

```python
queue = asyncio.Queue()
for item in genomes: queue.put_nowait(item)
for _ in self.workers: queue.put_nowait(None)   # ← сигнал «хватит»
```

Это **паттерн «пул задач + сентинелы»**:

- В очередь кладём все геномы.
- В конец — `None` для каждого воркера.
- Воркер берёт задачу, пока не встретит `None` → выходит.

```python
async def consume(worker):
    while True:
        item = await queue.get()
        if item is None: break
        genome_id, genome = item
        outcome = await worker.run_genome(genome)
        genome.fitness = outcome.fitness      # ← записываем в геном!
        results[genome_id] = outcome
        async with lock:
            completed += 1
            if completed % max(1, total // 10) == 0:
                print(f"оценено {completed}/{total} ...")
```

15 таких `consume` запускаются одновременно (`gather`). Кто освободился —
берёт следующего.

**🧠 Запомни:** очередь = «лоток с заданиями». 15 рабочих берут по одному.

**Зачем `lock`?** Два воркера могут одновременно увеличить `completed` —
получим гонку. `asyncio.Lock` пускает по одному.

---

### Блок 10. `TrainingLog` — CSV-журнал (строки 964–993)

```python
HEADERS = ["generation", "best_fitness", "mean_fitness", "best_score",
           "min_score", "max_score", "mean_score", "errors", "workers", "seconds"]
```

```python
def __init__(self, path):
    need_header = not os.path.exists(path)
    if os.path.exists(path):
        first_line = open(path).readline()
        existing = [h.strip() for h in first_line.split(",")]
        if existing != self.HEADERS:
            need_header = True          # старый формат → пишем новый заголовок
    self._handle = open(path, "a", newline="", encoding="utf-8")
    self._writer = csv.DictWriter(self._handle, fieldnames=self.HEADERS)
```

```python
def write(self, row):
    self._writer.writerow({k: row.get(k, "") for k in self.HEADERS})
    self._handle.flush()      # ← сразу на диск, чтобы не потерять при падении
```

**🧠 Запомни:** `flush()` = «не держи в буфере, запиши на диск сейчас же».

---

### Блок 11. Сохранение (строки 999–1030)

```python
def save_best(genome, config, args):
    pickle.dump(genome, open(args.model_file, "wb"))         # бинарно
    json.dump(network_to_js(genome, config),
              open(args.network_file, "w"))                   # читаемо
```

Два формата — два назначения:
- `.pkl` — для Python (продолжить обучение);
- `.json` — для людей и JS (посмотреть/использовать сеть).

```python
def save_elite(genomes, config, args, top_n=5):
    ranked = sorted(genomes, key=lambda item: item[1].fitness or -1e9,
                    reverse=True)
    for rank, (genome, result) in enumerate(ranked[:top_n], start=1):
        prefix = f"elite-{rank:02d}_score{result.score}_fit{result.fitness:.0f}"
```

**🧠 Запомни:** `key=...` — «по какому признаку сортировать»,
`reverse=True` — по убыванию.

`-1e9` — страховка: если `fitness is None`, ставим очень маленькое число.

---

### Блок 12. `train()` — главный цикл (строки 1033–1187)

#### 12.1 Подготовка

```python
config.pop_size = args.population
config.no_fitness_termination = True     # не останавливаться по порогу
```

```python
if args.resume:
    population = neat.Checkpointer.restore_checkpoint(args.resume, new_config=config)
else:
    population = neat.Population(config)
```

```python
population.add_reporter(neat.StdOutReporter(True))    # печать в консоль
population.add_reporter(neat.StatisticsReporter())   # статистика
population.add_reporter(neat.Checkpointer(
    args.checkpoint_every, filename_prefix="neat-checkpoint-"))
```

**🧠 Запомни:** reporter = «наблюдатель». Он не влияет на эволюцию, только
смотрит и записывает.

#### 12.2 Восстановление лучшего из прошлого запуска

```python
if os.path.exists(args.model_file):
    previous = load_genome(args.model_file)
    if previous.fitness is not None:
        best_ever = previous
```

Чтобы случайный неудачный запуск не «затёр» хороший результат.

#### 12.3 Основной цикл

```python
async with async_playwright() as playwright:
    async with AsyncEvaluator(playwright, config, args) as evaluator:
        for _ in range(args.generations):
```

##### Curriculum

```python
progress = min(1.0, generation / max(1, args.generations - 1))
args.episode_seconds += progress * (args.curriculum_max_seconds - args.episode_seconds)
for worker in evaluator.workers:
    worker.agent_cfg["seconds"] = args.episode_seconds
```

**Идея:** учим сначала на коротких эпизодах (30 с), потом удлиняем (60 с).
Как в школе: сначала простые задачи, потом сложные.

##### Сид трассы

```python
if args.route_seed is None:
    route_seed = random.randrange(1, 2**31 - 1)   # новый на поколение
else:
    route_seed = args.route_seed + generation    # воспроизводимо
evaluator.set_route_seed(route_seed)
```

##### Оценка

```python
genomes = list(population.population.items())
results = await evaluator.evaluate(genomes)
```

##### Статистика

```python
scored = [(population.population[gid], res) for gid, res in results.items()]
best_genome, best_result = max(scored, key=lambda item: item[0].fitness or -1e9)
mean_fitness = sum(g.fitness or 0.0 for g, _ in scored) / max(1, len(scored))
scores = [res.score for _, res in scored]
errors = sum(1 for _, res in scored if res.status in (...))
```

**🧠 Запомни:** `max(..., key=...)` — «взять элемент с максимальным fitness».

##### Сохранение лучшего

```python
improved = best_result.score > best_score_ever or (
    best_result.score == best_score_ever
    and (best_genome.fitness or 0) > (best_ever.fitness if best_ever else -1e9))
if best_ever is None or improved:
    best_ever = best_genome
    best_score_ever = best_result.score
    save_best(best_genome, config, args)
```

Сравниваем **по score**, а не по fitness. Почему? Fitness с бонусом
выживания может завысить «того, кто просто дожил», но играть не умеет.

##### Репродукция — сердце NEAT

```python
population.reporters.post_evaluate(config, population.population,
                                   population.species, best_genome)

population.population = population.reproduction.reproduce(
    config, population.species, config.pop_size, generation)

if not population.species.species:          # все вымерли
    population.reporters.complete_extinction()
    population.population = population.reproduction.create_new(
        config.genome_type, config.genome_config, config.pop_size)

population.species.speciate(config, population.population, generation)
population.reporters.end_generation(config, population.population, population.species)
population.generation += 1
```

Порядок **важен**:
1. `post_evaluate` — сообщить репортёрам результаты;
2. `reproduce` — создать новое поколение;
3. проверка вымирания — если никого не осталось, создать заново;
4. `speciate` — разбить на виды (похожие геномы);
5. `end_generation` — финальная статистика.

**🧠 Запомни:** мы **вручную** повторяем то, что делает `population.run()`,
потому что наша оценка асинхронная и своя.

#### 12.4 Финал

```python
except KeyboardInterrupt:
    print("\nПрервано пользователем.")
finally:
    log.close()                 # закроется всегда

if best_ever is not None:
    save_best(best_ever, config, args)

if args.dashboard and generation_history:
    generate_dashboard(args.log_file, "training_dashboard.html")
```

---

### Блок 13. `generate_dashboard()` (строки 1190–1298)

```python
with open(log_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        generations.append(int(row["generation"]))
        best_fitness.append(float(row["best_fitness"]))
        ...
```

Читает CSV → строит списки → подставляет в HTML-шаблон → пишет файл.
Графики рисует библиотека **Chart.js** (подключается с CDN).

**🧠 Запомни:** дашборд — это просто HTML-файл, который открывается в браузере.

---

### Блок 14. `play()` — показать игру (строки 1303–1333)

```python
genome = load_genome(args.model_file)
args.headless = False      # окно обязательно
args.live = True           # живой режим
args.episode_seconds = args.eval_seconds
args.max_frames = 10_000_000   # практически без лимита

async with async_playwright() as playwright:
    worker = BrowserWorker(0, playwright, config, args)
    await worker.start()
    try:
        result = await worker.run_genome(genome)
        print(f"Результат: ...")
    finally:
        await worker.close()
```

Один воркер, видимое окно, живой режим. Всё.

---

### Блок 15. CLI (строки 1339–1419)

```python
def parse_args():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--browsers", type=int, default=15)
    ...
    args = parser.parse_args()
    args.headless = not args.headed      # --headed → headless=False
    return args
```

**🧠 Запомни:** `store_true` = флаг-переключатель (есть/нет),
`type=int` = привести к числу, `default` = значение по умолчанию.

```python
def build_config(path):
    return neat.Config(
        neat.DefaultGenome,        # как хранить геном
        neat.DefaultReproduction,  # как размножать
        neat.DefaultSpeciesSet,    # как группировать в виды
        neat.DefaultStagnation,    # когда считать вид застойным
        path,                      # ini-файл
    )
```

**🧠 Запомни:** четыре «кирпича» NEAT: геном, размножение, виды, застой.

```python
def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(...)
    config = build_config(args.config)
    if args.eval:
        asyncio.run(play(args, config))
    else:
        asyncio.run(train(args, config))
```

**🧠 Запомни:** `asyncio.run(...)` — «запусти асинхронную функцию и дождись».

---

## 3. `evaluate.py` — проверка на новых трассах

```python
def build_args(opts):
    headed = opts.headed or opts.live
    return argparse.Namespace(
        browsers=1, headless=not headed,
        episode_seconds=opts.seconds, max_frames=opts.max_frames,
        clear_ms=900, poll_interval=0.15,
        route_seed=opts.seeds[0], live=opts.live)
```

Зачем собирать `Namespace` вручную? `BrowserWorker` ждёт объект с полями
`args.episode_seconds` и т.д. Здесь мы делаем его «совместимым».

```python
for seed in opts.seeds:
    worker.agent_cfg["seed"] = seed
    result = await worker.run_genome(genome)
    print(f"сид={seed}: ...")
```

**Смысл файла:** прогнать **один** геном на **нескольких невиданных** трассах.
Если на всех хорош — сеть действительно научилась, а не заучила одну карту.

**🧠 Запомни:** обучение = тренировка, `evaluate.py` = экзамен на новых билетах.

---

## 4. `neat_config.ini` — настройки эволюции

### `[NEAT]`

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `pop_size` | 150 | сколько особей (переопределяется `--population`) |
| `fitness_criterion` | max | максимизируем fitness |
| `fitness_threshold` | 1000000 | недостижимо → не останавливаться |
| `reset_on_extinction` | True | все вымерли → начать заново |

### `[DefaultGenome]`

| Параметр | Смысл |
|----------|-------|
| `num_inputs = 19` | глаза |
| `num_outputs = 2` | прыжок + присед |
| `feed_forward = True` | без циклов (только вперёд) |
| `initial_connection = full` | в начале все со всеми |
| `conn_add_prob = 0.5` | шанс добавить связь |
| `node_add_prob = 0.2` | шанс добавить нейрон |
| `activation_options = tanh relu sigmoid` | какие функции разрешены |
| `weight_mutate_rate = 0.8` | 80% связей мутируют вес |

**🧠 Запомни:** `*_prob` — вероятность мутации, `*_rate` — частота,
`*_power` — сила изменения.

### `[DefaultSpeciesSet]`

```ini
compatibility_threshold = 3.0
```

Насколько геномы должны быть похожи, чтобы считаться одним видом.
Меньше → больше видов; больше → все в одном.

### `[DefaultStagnation]`

```ini
max_stagnation = 20     # вид без улучшений 20 поколений → не размножается
species_elitism = 2     # 2 лучших в виде защищены
```

### `[DefaultReproduction]`

```ini
elitism = 2             # 2 лучших генома переходят без изменений
survival_threshold = 0.2  # размножается только топ-20% вида
```

**🧠 Запомни:** элитизм = «лучшие не умирают», это стабилизирует обучение.

---

## 5. Шпаргалка: 12 блоков за 5 минут

| № | Блок | Суть в одну строку |
|---|------|--------------------|
| 1 | Импорты | `pickle` — объект в файл, `json` — объект в текст |
| 2 | Геометрия | размеры из игры, `y` растёт вниз |
| 3 | `FEATURE_NAMES` | контракт на 19 входов |
| 4 | `AGENT_JS` | вся игра + сеть живут в браузере |
| 5 | `network_to_js` | Python-сеть → JSON для JS |
| 6 | `EpisodeResult` | коробка с результатом |
| 7 | `make_result` | score → fitness (с бонусами и штрафами) |
| 8 | `BrowserWorker` | один браузер, 3 попытки, не падает |
| 9 | `AsyncEvaluator` | очередь задач на 15 воркеров |
| 10 | `TrainingLog` | CSV + `flush()` |
| 11 | `save_best/elite` | pickle для Python, json для людей |
| 12 | `train` | цикл: оценили → отобрали → размножили |

---

## 6. Пять идей, которые надо запомнить навсегда

1. **Python — тренер, браузер — спортсмен.** Python не играет, он оценивает.
2. **Сеть копируется внутрь страницы.** Иначе 60 запросов в секунду — слишком
   медленно.
3. **Часы замораживаются.** `performance.now = () => clock` → эпизод считается
   в 40 раз быстрее и воспроизводимо.
4. **Сид делает сравнение честным.** Один сид → одна трасса → все в равных
   условиях.
5. **Fitness ≠ score.** Score — очки игры, fitness — оценка для эволюции
   (очки + бонусы − штрафы).

---

## 7. Как самому проверить, что понял

Попробуй ответить, не подглядывая:

1. Почему `network_to_js` вызывается **до** `simulate`, а не наоборот?
2. Что случится, если убрать `installSeededRandom`?
3. Зачем в `evaluate` кладут `None` в очередь?
4. Почему `_force_restart` закрывает браузер, а `_recover` — нет?
5. Что будет, если в `make_result` убрать `SURVIVAL_BONUS`?
6. Почему `save_best` пишет **два** файла?
7. Зачем `async with` у `AsyncEvaluator`?
8. Что делает `flush()` и почему без него можно потерять данные?

Ответы ищи в разделах выше — по одному вопросу на блок.
