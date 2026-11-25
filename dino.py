"""Dino Runner: обучение нейросети NEAT в 15 браузерах Chromium одновременно.

Что изменено по сравнению с Q-learning версией:
  * алгоритм — NEAT (нейроэволюция), а не таблица Q-значений;
  * полностью асинхронный Playwright (``async_playwright``);
  * пул из N независимых браузеров Chromium (по умолчанию 15), каждый геном
    играет свой эпизод в своём браузере, оценки идут конкурентно;
  * физика и коллизии не дублируются: решение принимается прямо в JS-контексте
    страницы, а столкновение ловит сама игра (``runner.crashed``);
  * сеть переносится в JavaScript, поэтому на каждый игровой кадр не тратится
    сетевой round-trip Playwright;
  * лучший геном сохраняется (pickle + JSON-описание сети), есть чекпоинты
    поколений и CSV-журнал.

Среда (значения получены из исходника ``index.js`` игры T-Rex Runner):
  * канвас 600x150, земля: дино стоит на y=93, высота 47, присед 25;
  * малый кактус 17x35 @ y=105, крупный 25x50 @ y=90;
  * птеродактиль 46x40, высоты y ∈ {100, 75, 50}, появляется при speed >= 8.5;
  * скорость 6 → 13 (ACCELERATION 0.001/кадр), счёт = distanceRan * 0.025.

Запуск:
    python dino.py                      # обучение, 15 браузеров
    python dino.py --browsers 15 --generations 30
    python dino.py --headed             # с видимыми окнами браузеров
    python dino.py --eval               # показать игру лучшим геномом
    python dino.py --resume neat-checkpoint-10
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import neat
from neat.graphs import feed_forward_layers
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

GAME_URL = "https://wayou.github.io/t-rex-runner/"
NEAT_CONFIG = "neat_config.ini"

# ---------------------------------------------------------------------------
# Геометрия мира (из исходника игры)
# ---------------------------------------------------------------------------
CANVAS_W = 600
CANVAS_H = 150
GROUND_Y = 93          # верхняя граница дино, когда он на земле
DINO_H = 47
DINO_H_DUCK = 25
DINO_W = 44
MAX_SPEED = 13
START_SPEED = 6
SCORE_COEF = 0.025

# ---------------------------------------------------------------------------
# Входные признаки сети (порядок обязан совпадать с buildInputs() в JS)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "dx1_norm",         # расстояние до первого препятствия
    "dx2_norm",         # расстояние до второго препятствия
    "has_second",       # есть ли второе препятствие
    "is_ptero",         # первое препятствие — птеродактиль
    "ptero_y_norm",     # высота птицы (100 / 75 / 50)
    "ground_run_collision",
    "ground_duck_collision",
    "obs_height_norm",
    "obs_width_norm",
    "obs_top_norm",
    "obs_bottom_norm",
    "gap_after_norm",   # зазор справа от препятствия до края дино
    "dino_airborne",    # высота дино над землёй
    "jump_velocity",
    "jumping",
    "ducking",
    "speed_drop",
    "speed_norm",
    "apex_collision",    # столкновение с препятствием на вершине прыжка
]
NUM_INPUTS = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# JS-агент, который живёт в странице: сеть + сбор признаков + управление
# ---------------------------------------------------------------------------
AGENT_JS = r"""
(() => {
  if (window.__dino) { return; }

  const DEFAULTS = {
    canvasWidth: 600,
    seconds: 30,
    maxFrames: 4000,
    clearMs: 900,
    seed: 1,
    render: false,
    liveSpeed: 1,
    pteroReflex: true,
    trace: 0
  };

  const FRAME_MS = 1000 / 60;

  const ACT = {
    sigmoid: z => 1.0 / (1.0 + Math.exp(-Math.max(-60, Math.min(60, 5.0 * z)))),
    tanh: z => Math.tanh(Math.max(-60, Math.min(60, 2.5 * z))),
    sin: z => Math.sin(Math.max(-60, Math.min(60, 5.0 * z))),
    gauss: z => { z = Math.max(-3.4, Math.min(3.4, z)); return Math.exp(-5.0 * z * z); },
    relu: z => (z > 0 ? z : 0),
    elu: z => (z > 0 ? z : Math.exp(z) - 1),
    lelu: z => (z > 0 ? z : 0.005 * z),
    selu: z => (z > 0 ? 1.0507009873554805 * z
                      : 1.0507009873554805 * 1.6732632423543772 * (Math.exp(z) - 1)),
    softplus: z => 0.2 * Math.log(1 + Math.exp(Math.max(-60, Math.min(60, 5.0 * z)))),
    identity: z => z,
    clamped: z => Math.max(-1, Math.min(1, z)),
    inv: z => (z === 0 ? 0 : 1.0 / z),
    log: z => Math.log(Math.max(1e-7, z)),
    exp: z => Math.exp(Math.max(-60, Math.min(60, z))),
    abs: z => Math.abs(z),
    hat: z => Math.max(0, 1 - Math.abs(z)),
    square: z => z * z,
    cube: z => z * z * z
  };

  const AGG = {
    sum: xs => xs.reduce((a, b) => a + b, 0),
    mean: xs => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0),
    max: xs => (xs.length ? Math.max.apply(null, xs) : 0),
    min: xs => (xs.length ? Math.min.apply(null, xs) : 0),
    product: xs => xs.reduce((a, b) => a * b, 1),
    // ВАЖНО: neat-python возвращает САМ элемент с наибольшим модулем
    // (max(x, key=abs)), а не его модуль. Раньше здесь возвращался |x| —
    // и сеть в браузере считала не то же самое, что оценивал геном в Python.
    maxabs: xs => {
      if (!xs.length) { return 0; }
      return xs.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a));
    },
    median: xs => {
      if (!xs.length) { return 0; }
      const s = xs.slice().sort((a, b) => a - b);
      const m = s.length >> 1;
      return (s.length % 2) ? s[m] : 0.5 * (s[m - 1] + s[m]);
    }
  };

  let NET = null;
  let CFG = Object.assign({}, DEFAULTS);
  let state = 'idle';
  let result = null;
  let liveRaf = 0;
  let realNowRef = null;
  let run = null;
  let seed = 1;
  let traceBuf = [];

  const clamp = (x, lo, hi) => (x < lo ? lo : (x > hi ? hi : x));
  const norm = (x, scale) => clamp(x / scale, -2, 2);

  // Детерминированный ГПСЧ (mulberry32). Все геномы поколения получают одну
  // и ту же трассу, иначе отбор превращается в лотерею по расстояниям/птицам.
  function installSeededRandom() {
    let s = seed >>> 0;
    Math.random = function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function activate(inputs) {
    const values = {};
    for (let i = 0; i < NET.inputs.length; i++) {
      values[NET.inputs[i]] = inputs[i] || 0;
    }
    for (let n = 0; n < NET.evals.length; n++) {
      const node = NET.evals[n];
      const xs = [];
      for (let k = 0; k < node.links.length; k++) {
        const inode = node.links[k][0];
        const w = node.links[k][1];
        xs.push((values[inode] || 0) * w);
      }
      const agg = AGG[node.agg] || AGG.sum;
      const act = ACT[node.act] || ACT.tanh;
      values[node.node] = act(node.bias + node.response * agg(xs));
    }
    return NET.outputs.map(k => values[k]);
  }

  // --- Физика дино для классификации препятствий --------------------------
  // Смещения коллизионных боксов взяты из game_source.js (Trex.collisionBoxes).
  // Они заданы относительно точки (xPos+1, yPos+1), поэтому при проверке
  // прибавляем по единице — ровно как делает checkForCollision().
  const TREX_BOXES = {
    RUN: [
      [22, 0, 17, 16], [1, 18, 30, 9], [10, 35, 14, 8],
      [1, 24, 29, 5], [5, 30, 21, 4], [9, 34, 15, 4]
    ],
    DUCK: [[1, 18, 55, 25]]
  };

  // Максимальная высота прыжка при данной скорости. Повторяет дискретную
  // физику Trex.updateJump: y += round(v); v += GRAVITY, с обрезкой
  // endJump при yPos < MAX_JUMP_HEIGHT(30), если достигнут MIN_JUMP_HEIGHT(63).
  const apexCache = new Map();
  function jumpApex(speed) {
    const key = Math.round(speed * 100);
    if (apexCache.has(key)) { return apexCache.get(key); }
    const groundY = 93;
    let y = groundY;
    let v = -10 - speed / 10;       // INIITAL_JUMP_VELOCITY - speed/10
    let best = groundY;
    for (let i = 0; i < 400; i++) {
      y += Math.round(v);
      if (y < best) { best = y; }
      v += 0.6;                     // GRAVITY
      // endJump: yPos < MAX_JUMP_HEIGHT(30) и летим вверх → обрезка до -5
      if (y < 30 && v < -5) { v = -5; }
      if (y >= groundY) { break; }
    }
    const height = groundY - best;
    apexCache.set(key, height);
    return height;
  }

  // Пересекается ли динозавр в заданной позе с препятствием, ЕСЛИ БЫ оно
  // стояло вплотную к нему. Именно так классифицируется препятствие:
  // «бег врезается / присед врезается / прыжок врезается». Проверять в
  // текущей дальней позиции бессмысленно — там пересечения нет ни с чем.
  function wouldHit(obstacle, dinoX, dinoY, ducking) {
    const boxes = ducking ? TREX_BOXES.DUCK : TREX_BOXES.RUN;
    const oboxes = obstacle.collisionBoxes;
    if (!oboxes || !oboxes.length) { return false; }
    const ox0 = dinoX + 1;   // препятствие «приставлено» к дино
    const oy0 = obstacle.yPos + 1;
    for (let i = 0; i < boxes.length; i++) {
      const t = boxes[i];
      const tx = dinoX + 1 + t[0];
      const ty = dinoY + 1 + t[1];
      for (let j = 0; j < oboxes.length; j++) {
        const o = oboxes[j];
        const bx = ox0 + o.x;
        const by = oy0 + o.y;
        if (tx < bx + o.width && tx + t[2] > bx &&
            ty < by + o.height && ty + t[3] > by) {
          return true;
        }
      }
    }
    return false;
  }

  // Рефлекс на птеродактиля. Сеть выучила только прыжки и на любой птице
  // прыгает — в том числе на высоко летящей, где прыжок вообще не нужен.
  // Геометрия хитбоксов (из game_source.js):
  //   бег дино    94..137
  //   присед дино 112..137
  //   птица y=50  59..78   (высоко)  — бег и присед проходят, прыжок лезет в неё
  //   птица y=75  84..103            — бег задевает, присед проходит
  //   птица y=100 109..128 (низко)   — задевает и присед, нужен прыжок
  // Поэтому под двумя первыми высотами приседаем, а низкую птицу оставляем
  // сети: её надо перепрыгивать, и геном это уже умеет.
  const REFLEX_DUCK_LEAD = 350;   // px до дино, когда уже поздно прыгать
  const DINO_W_RUN = 44;
  const DINO_W_DUCK = 59;

  function pteroReflex(obstacles, dino, outputs) {
    if (!CFG.pteroReflex || !dino) { return null; }
    // Если сеть уже решила прыгать — не мешаем. Присед, отменивший прыжок,
    // делает его поздним, и дальше динозавр приземляется на следующий кактус.
    if (outputs[0] > 0.5) { return null; }
    // В воздухе приседать нельзя: setDuck во время прыжка сбивает его физику.
    if (dino.jumping) { return null; }
    const groundY = dino.groundYPos;
    const dinoW = dino.ducking ? DINO_W_DUCK : DINO_W_RUN;
    let ptero = null;
    for (const obstacle of obstacles) {
      if (!obstacle || !obstacle.typeConfig) { continue; }
      if (obstacle.typeConfig.type !== 'PTERODACTYL') { continue; }
      const frontGap = obstacle.xPos - (dino.xPos + dinoW);
      const rearGap = obstacle.xPos + obstacle.width - dino.xPos;
      if (rearGap <= 0 || frontGap > REFLEX_DUCK_LEAD) { continue; }
      ptero = obstacle;
      break;
    }
    if (!ptero) { return null; }

    // Приседаем только под ВЫСОКО летящей птицей (y=50), которая и так
    // пролетает над бегущим динозавром. Там присед ничего не ломает физически:
    // столкновения нет ни в беге, ни в приседе, меняется только поза.
    // Птицу, которая задевает бегущего, оставляем сети: её тайминг прыжков
    // выверен, и трогать его нельзя.
    if (wouldHit(ptero, dino.xPos, groundY, false)) { return null; }

    // Присед шире бегового хитбокса (59 против 44). Если рядом впереди другое
    // препятствие, не приседаем: широкий бокс может зацепить его раньше.
    for (const obstacle of obstacles) {
      if (!obstacle || obstacle === ptero) { continue; }
      if (obstacle.xPos + obstacle.width < dino.xPos) { continue; }
      if (obstacle.xPos - dino.xPos < 240) { return null; }
    }
    return [0, 1];
  }

  function buildInputs(runner, standingView) {
    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
    const w = CFG.canvasWidth || 600;
    const obstacle = obstacles.length ? obstacles[0] : null;
    const speed = runner.currentSpeed || 6;
    const f = new Array(19).fill(0);

    // standingView: сеть всегда видит дино стоящим. Присед включает отдельный
    // защитный слой (pteroReflex), и его не надо подмешивать во входы обученной
    // политики: иначе геном, который никогда не приседал, получает непривычное
    // состояние и срывается в прыжки «в пустоту».
    const dinoW = (!standingView && dino.ducking) ? 59 : 44;
    const dinoRight = dino.xPos + dinoW;
    // Расстояние до препятствия — от переднего края дино. Нормируем по
    // рабочему окну реакции (~160 px): иначе 400 и 120 px дают почти
    // одинаковые 0.67 и 0.20.
    const dnorm = x => clamp(x / 160, -1, 3);
    const edge = ob => ob.xPos - dinoRight;

    if (!obstacle) {
      f[0] = 3; f[1] = 3; f[11] = 3;
    } else {
      const dx1 = edge(obstacle);
      f[0] = dnorm(dx1);
      f[1] = obstacles.length > 1 ? dnorm(edge(obstacles[1])) : 3;
      f[2] = obstacles.length > 1 ? 1 : 0;

      const type = obstacle.typeConfig ? obstacle.typeConfig.type : '';
      const oh = obstacle.typeConfig ? obstacle.typeConfig.height : 0;
      const oy = obstacle.yPos || 0;
      const isPtero = type === 'PTERODACTYL';
      const groundY = dino.groundYPos;

      f[3] = isPtero ? 1 : 0;
      f[4] = clamp(oy / 150, -1, 1);

      // Физически посчитанные исходы. Именно этих сигналов не хватало:
      // без них сеть не могла отличить смертельное препятствие от того,
      // которое само пролетает над динозавром (птеродактиль на y=50).
      f[5] = wouldHit(obstacle, dino.xPos, groundY, false) ? 1 : 0;  // бег врезается
      f[6] = wouldHit(obstacle, dino.xPos, groundY, true) ? 1 : 0;   // присед врезается
      const apexY = groundY - jumpApex(speed);
      f[18] = wouldHit(obstacle, dino.xPos, apexY, false) ? 1 : 0;   // прыжок врезается

      f[7] = clamp(oh / 50, 0, 2);
      f[8] = clamp(obstacle.width / 50, 0, 2);
      f[9] = clamp(oy / 150, -1, 1);
      f[10] = clamp((oy + oh) / 150, -1, 1);
      f[11] = dnorm(dx1 - obstacle.width);
    }

    const airborne = Math.max(0, dino.groundYPos - dino.yPos);
    const jumpVelocity = dino.jumpVelocity;
    f[12] = clamp(airborne / 50, 0, 1);
    f[13] = clamp((typeof jumpVelocity === 'number' ? jumpVelocity : 0) / 15, -2, 2);
    f[14] = dino.jumping ? 1 : 0;
    f[15] = (!standingView && dino.ducking) ? 1 : 0;
    f[16] = dino.speedDrop ? 1 : 0;
    f[17] = clamp((speed - 6) / 7, 0, 1);
    return f;
  }

  function apply(runner, jumpValue, duckValue) {
    const dino = runner.tRex;
    const doJump = jumpValue > 0.5;
    const doDuck = !doJump && duckValue > 0.5;
    try {
      if (doJump) {
        if (dino.ducking) { dino.setDuck(false); }
        if (!dino.jumping) {
          dino.startJump(runner.currentSpeed);
          run.jumps += 1;
          // Полезность прыжка определяется физически: приземлился ли он так,
          // что под динозавром успело пройти препятствие (см. trackJump).
          run.jumpPending = true;
          run.jumpSafe = false;
        }
        return;
      }
      if (doDuck) {
        if (dino.jumping) { dino.setSpeedDrop(); }
        // ВАЖНО: Trex.setDuck(true) при уже активном приседе ВЫКЛЮЧАЕТ его
        // (так написан оригинальный код игры: ветка else if status==DUCKING).
        // Поэтому вызываем только на переходе, иначе получается дребезг
        // «присед/встал» каждый кадр и динозавр гибнет о птицу.
        else if (!dino.ducking) { dino.setDuck(true); }
        return;
      }
      if (dino.ducking) { dino.setDuck(false); }
    } catch (err) {
      // Игра может в этот момент быть в состоянии рестарта — пропускаем кадр.
    }
  }

  // Прыжок считается полезным, если, пока динозавр был в воздухе, под ним
  // реально прошло препятствие. Прыжок «в пустоту» (приземлился раньше, чем
  // препятствие доехало) — бесполезный. Это не зависит от скорости.
  function trackJump(runner) {
    if (!run || !run.jumpPending) { return; }
    const dino = runner.tRex;
    if (!dino) { run.jumpPending = false; return; }
    if (!dino.jumping) {
      run.jumpPending = false;
      if (!run.jumpSafe) { run.useless += 1; }
      return;
    }
    const dinoLeft = dino.xPos;
    const dinoRight = dino.xPos + (dino.ducking ? 59 : 44);
    const obs = (runner.horizon && runner.horizon.obstacles) || [];
    for (let i = 0; i < obs.length; i++) {
      const o = obs[i];
      if (o.xPos < dinoRight && (o.xPos + o.width) > dinoLeft) {
        run.jumpSafe = true;
        break;
      }
    }
  }

  // Останавливает наш цикл показа. Штатный Runner.scheduleNextUpdate отключён,
  // поэтому единственный источник кадров — этот цикл.
  function stopLiveLoop() {
    if (!liveRaf) { return; }
    clearTimeout(liveRaf);
    liveRaf = 0;
  }

  // Темп держим на setTimeout, а не на requestAnimationFrame: Chromium режет
  // rAF до 1-2 тиков в секунду, если окно перекрыто другим окном или экран
  // погас, и тогда игра ползёт в разы медленнее реального времени. Сколько
  // кадров проиграть, решает не тик, а накопленное реальное время.
  function scheduleLive(fn) {
    liveRaf = setTimeout(fn, 8);
  }

  function finish(status) {
    if (state !== 'running') { return; }
    stopLiveLoop();
    const runner = window.Runner && window.Runner.instance_;
    if (run && run.jumpPending && !run.jumpSafe) {
      // Прыжок остался «в воздухе» и не накрыл препятствие.
      run.useless += 1;
    }
    if (run) { run.jumpPending = false; }
    if (realNowRef) { performance.now = realNowRef; realNowRef = null; }
    state = 'done';
    result = snapshotResult(status, runner);
    updateLiveOverlay(runner, status);
    try { if (runner) { runner.stop(); } } catch (err) { /* noop */ }
  }

  // Готовит страницу к показу: убирает служебный текст Chromium
  // («Press Space to start», иконку офлайна) и увеличивает игровое поле,
  // иначе сцена 600x150 теряется в окне и её плохо видно.
  // Вызывается ОДИН раз при старте: пересчёт стилей каждый кадр замедлял бег.
  let stageReady = false;
  function prepareStage(runner) {
    if (!runner || !runner.canvas) { return; }
    const canvas = runner.canvas;
    if (!stageReady) {
      stageReady = true;
      document.body.style.background = '#f7f7f7';
      document.body.style.overflow = 'hidden';
      document.documentElement.style.overflow = 'hidden';
      document.documentElement.style.background = '#f7f7f7';

      // Служебные элементы страницы офлайна Chromium: заголовок «Press Space
      // to start» (он лежит внутри #messageBox, а не в .interstitial-wrapper),
      // иконка офлайна, снекбар и подсказки.
      const junk = document.querySelectorAll(
        '#messageBox, h1, .icon-offline, #offline-resources, .snackbar, #main-content'
      );
      for (let i = 0; i < junk.length; i++) {
        junk[i].style.setProperty('display', 'none', 'important');
      }

      const host = runner.containerEl || canvas.parentElement;
      if (host) {
        // Контейнер игры по умолчанию шириной 44 px и с overflow:hidden —
        // из-за этого canvas обрезался. Позиционируем сам canvas.
        host.style.position = 'static';
        host.style.transform = 'none';
        host.style.overflow = 'visible';
        host.style.width = 'auto';
        host.style.height = 'auto';
        host.style.margin = '0';
        host.style.padding = '0';
      }

      // Сама игра периодически переписывает инлайновые width/height у canvas
      // (updateCanvasScaling), поэтому масштаб задаём правилом с !important:
      // оно перебивает инлайновый стиль и держит поле крупным.
      // Отдельно снимаем max-width/max-height: в CSS страницы есть правило
      // `.offline .runner-canvas { max-width: 600px }`, которое обрезало
      // увеличенный canvas по ширине.
      const style = document.createElement('style');
      style.id = '__dino_demo_stage';
      style.textContent = [
        'body { margin:0; background:#f7f7f7; overflow:hidden; }',
        'canvas.runner-canvas {',
        '  display:block !important;',
        '  visibility:visible !important;',
        '  opacity:1 !important;',
        '  position:fixed !important;',
        '  left:50% !important;',
        '  top:50% !important;',
        '  transform:translate(-50%,-50%) !important;',
        '  z-index:1 !important;',
        '  image-rendering:pixelated !important;',
        '  max-width:none !important;',
        '  max-height:none !important;',
        '  min-width:0 !important;',
        '  min-height:0 !important;',
        '}'
      ].join('\n');
      document.head.appendChild(style);
    }

    // Масштаб пересчитываем только при изменении размера окна.
    const w = runner.dimensions ? runner.dimensions.WIDTH : 600;
    const h = runner.dimensions ? runner.dimensions.HEIGHT : 150;
    const key = window.innerWidth + 'x' + window.innerHeight + ':' + w + 'x' + h;
    if (canvas.__stageKey === key) { return; }
    canvas.__stageKey = key;
    const scale = Math.max(1, Math.min(
      (window.innerWidth * 0.92) / w,
      (window.innerHeight * 0.78) / h
    ));
    const rule = document.getElementById('__dino_demo_size');
    const css = 'canvas.runner-canvas {' +
      ' width:' + Math.round(w * scale) + 'px !important;' +
      ' height:' + Math.round(h * scale) + 'px !important; }';
    if (rule) { rule.textContent = css; }
    else {
      const el = document.createElement('style');
      el.id = '__dino_demo_size';
      el.textContent = css;
      document.head.appendChild(el);
    }
  }

  // Крупный счётчик для демонстрации. Штатный DistanceMeter игры маленький
  // и на некоторых масштабах Chromium почти не виден, поэтому дублируем его
  // поверх canvas обычным DOM-текстом.
  function ensureLiveOverlay(runner) {
    if (!runner || !runner.canvas) { return null; }
    prepareStage(runner);
    let panel = document.getElementById('__dino_demo_overlay');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = '__dino_demo_overlay';
      panel.style.cssText = [
        'position:fixed', 'top:18px', 'right:24px', 'z-index:99999',
        'padding:12px 20px', 'border-radius:10px',
        'background:rgba(255,255,255,.94)', 'color:#222',
        'font:700 28px Arial,sans-serif', 'box-shadow:0 2px 12px rgba(0,0,0,.18)',
        'min-width:210px', 'text-align:center', 'pointer-events:none'
      ].join(';');
      document.body.appendChild(panel);
    }
    return panel;
  }

  let overlayPanel = null;

  function updateLiveOverlay(runner, status) {
    if (!runner) { return; }
    if (!overlayPanel) {
      overlayPanel = ensureLiveOverlay(runner);
      if (!overlayPanel) { return; }
    }
    const score = Math.round((runner.distanceRan || 0) * 0.025);
    const text = 'SCORE: <b>' + score + '</b>' +
      (status ? '<div style="font-size:15px;margin-top:4px">' + status + '</div>' : '');
    if (overlayPanel.__html !== text) {
      overlayPanel.__html = text;
      overlayPanel.innerHTML = text;
    }
  }

  // Приводит игру в исходное состояние, не запуская её собственный цикл.
  // Игровые часы берутся из замороженного performance.now, поэтому вся физика
  // двигается только нашим шагом.
  function resetGame(runner, clock, noRender) {
    runner.stop();
    runner.config.CLEAR_TIME = CFG.clearMs;
    runner.crashed = false;
    runner.playing = true;
    runner.paused = false;
    runner.activated = true;
    runner.playingIntro = false;
    runner.distanceRan = 0;
    runner.runningTime = 0;
    runner.invertTimer = 0;
    runner.setSpeed(runner.config.SPEED);
    if (runner.tRex) {
      runner.tRex.playingIntro = false;
      runner.tRex.jumpCount = 0;
    }
    if (runner.containerEl) { runner.containerEl.style.webkitAnimation = ''; }
    runner.raqId = 0;
    runner.time = clock;
    runner.soundFx = runner.soundFx || {};
    runner.playSound = function () {};
    runner.scheduleNextUpdate = function () { this.updatePending = false; };
    // Игра ставит себя на паузу, когда окно теряет фокус. Для показа это
    // лишнее: учитель может переключиться на другое окно, а бег должен идти.
    runner.onVisibilityChange = function () {};
    runner.highestScore = 0;

    // Полный сброс горизонта: reset() самой игры НЕ чистит clouds и
    // obstacleHistory, а облака тянут Math.random() и сдвигают поток ГПСЧ,
    // из-за чего трасса получалась разной от прогона к прогону.
    const horizon = runner.horizon;
    if (horizon) {
      horizon.obstacles = [];
      horizon.obstacleHistory = [];
      horizon.clouds = [];
      horizon.runningTime = 0;
      horizon.horizonOffsets = [0, 0];
    }
    if (runner.distanceMeter) {
      runner.distanceMeter.acheivement = false;
    }

    // Ночной режим игры при каждой смене фазы заново расставляет звёзды через
    // Math.random() — тот же генератор, что создаёт препятствия. Поэтому
    // включённая отрисовка меняла саму трассу (18889 в headless против 3637
    // в видимом окне). Отключаем смену дня/ночи: это только фон, а трасса
    // остаётся ровно той, на которой геном обучался.
    runner.invert = function () {};
    if (noRender) {
      runner.clearCanvas = function () {};
      const ctx = runner.canvasCtx;
      if (ctx && ctx.drawImage) { ctx.drawImage = function () {}; }
    }
    // В оригинальной игре дино въезжает на START_X_POS=50 во время intro,
    // и xPos после рестарта сохраняется. Мы intro не проигрываем, поэтому
    // ставим позицию руками — иначе дино стоит вплотную к краю (x=0) и у
    // агента на 50 px меньше времени на реакцию, чем у настоящей игры.
    if (runner.tRex) {
      runner.tRex.xPos = runner.tRex.config.START_X_POS || 50;
    }
    runner.restart();
  }

  function snapshotResult(status, runner) {
    const payload = {
      status: status,
      score: Math.round((runner.distanceRan || 0) * 0.025),
      frames: run.frames,
      seconds: Math.round(run.frames * FRAME_MS / 10) / 100,
      crashed: !!runner.crashed,
      jumps: run.jumps,
      useless: run.useless,
      ducks: run.ducks
    };
    if (CFG.trace && traceBuf.length) {
      payload.trace = traceBuf.slice(-CFG.trace);
    }
    return payload;
  }

  function newRun() {
    return {
      frames: 0, jumps: 0, useless: 0, ducks: 0,
      jumpPending: false, jumpSafe: false
    };
  }

  // --- Диагностика: кольцевой буфер последних кадров -----------------------
  // Включается CFG.trace. Хранит состояние за мгновение до решения сети, чтобы
  // после падения было видно, ЧТО именно не распознал геном: дистанцию до
  // препятствия, его высоту/тип, положение дино и выход сети.
  const TRACE_MAX = 400;

  function resetTrace() {
    traceBuf = [];
  }

  function recordFrame(runner, outputs) {
    if (!CFG.trace) { return; }
    const dino = runner.tRex;
    const obs = (runner.horizon && runner.horizon.obstacles) || [];
    const o = obs.length ? obs[0] : null;
    const round = (v, n) => (v === null || v === undefined ? null : Math.round(v * n) / n);
    traceBuf.push({
      f: run.frames,
      ox: o ? Math.round(o.xPos) : null,
      oy: o ? o.yPos : null,
      ow: o ? o.width : null,
      oh: o && o.typeConfig ? o.typeConfig.height : null,
      ot: o && o.typeConfig ? o.typeConfig.type.slice(0, 4) : '',
      dx: o && dino ? Math.round(o.xPos - dino.xPos) : null,
      dy: dino ? Math.round(dino.yPos) : null,
      gr: dino ? (dino.jumping ? 1 : 0) : 0,
      dk: dino ? (dino.ducking ? 1 : 0) : 0,
      sp: round(runner.currentSpeed, 100),
      jv: outputs ? round(outputs[0], 100) : null,
      dv: outputs ? round(outputs[1], 100) : null
    });
    if (traceBuf.length > TRACE_MAX) { traceBuf.shift(); }
  }

  // ---------------------------------------------------------------------
  // Быстрый детерминированный прогон эпизода.
  //
  // Часы страницы замораживаются (performance.now возвращает виртуальное
  // время), requestAnimationFrame не используется — цикл сам двигает игру
  // шагом ровно 1/60 c. Поэтому физика, расстояния и коллизии полностью
  // воспроизводимы, а эпизод считается за десятки миллисекунд вместо
  // десятков секунд реального времени.
  // ---------------------------------------------------------------------
  function decide(runner) {
    const outputs = activate(buildInputs(runner, true));
    const obs = (runner.horizon && runner.horizon.obstacles) || [];
    const reflex = pteroReflex(obs, runner.tRex, outputs);
    return reflex || outputs;
  }

  function simulate() {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner || !window.Runner.imageSprite || !NET) {
      return emptyResult('NO_RUNNER');
    }
    installSeededRandom();

    const realNow = performance.now.bind(performance);
    let clock = 1000;
    performance.now = () => clock;

    try {
      resetGame(runner, clock, !CFG.render);
    } catch (err) {
      performance.now = realNow;
      return emptyResult('ERROR', String(err));
    }

    run = newRun();
    state = 'running';
    result = null;
    resetTrace();

    const maxByTime = CFG.seconds > 0 ? Math.ceil(CFG.seconds * 60) : Infinity;
    const maxByFrames = CFG.maxFrames > 0 ? CFG.maxFrames : Infinity;
    // Жёсткая страховка: если оба лимита отключены, эпизод может крутиться
    // вечно и подвесить браузер. 260 000 кадров ≈ 72 минуты игры ≈ 84 000 очков.
    const SAFETY_FRAMES = 260000;
    const frameLimit = Math.min(maxByTime, maxByFrames, SAFETY_FRAMES);
    let status = 'FRAMES';

    try {
      for (let i = 0; i < frameLimit; i++) {
        if (runner.crashed) { status = 'DEAD'; break; }
        const outputs = decide(runner);
        recordFrame(runner, outputs);
        apply(runner, outputs[0], outputs[1]);
        clock += FRAME_MS;
        runner.update();
        trackJump(runner);
        run.frames = i + 1;
        if (runner.tRex && runner.tRex.ducking) { run.ducks += 1; }
      }
      if (runner.crashed) { status = 'DEAD'; }
      trackJump(runner);
    } finally {
      performance.now = realNow;
    }

    state = 'done';
    result = snapshotResult(status, runner);
    try { runner.stop(); } catch (err) { /* noop */ }
    return { state: state, result: result };
  }

  function emptyResult(status, error) {
    state = 'done';
    result = {
      status: status, score: 0, frames: 0, seconds: 0,
      crashed: false, jumps: 0, useless: 0, ducks: 0
    };
    if (error) { result.error = error; }
    return { state: state, result: result };
  }

  function init(payload) {
    NET = (payload && payload.net) || null;
    CFG = Object.assign({}, DEFAULTS, (payload && payload.cfg) || {});
    seed = (CFG.seed | 0) >>> 0;
    return { ok: !!NET };
  }

  // Живой режим (для --eval): игра рисуется в реальном времени, но физика
  // двигается ТЕМ ЖЕ фиксированным шагом 1/60 c, что и в simulate(). Иначе
  // requestAnimationFrame отдаёт плавающий dt (троттлинг окна, джиттер), прыжок
  // считается по другой траектории и обученный геном «не узнаёт» ситуацию.
  // rAF здесь нужен только для темпа и отрисовки.
  function startLive() {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner || !window.Runner.imageSprite || !NET) {
      return { ok: false, error: 'NOT_READY' };
    }
    if (liveRaf) { stopLiveLoop(); }

    installSeededRandom();
    ensureLiveOverlay(runner);

    const realNow = performance.now.bind(performance);
    let clock = 1000;
    performance.now = () => clock;
    realNowRef = realNow;

    try {
      resetGame(runner, clock, false);
    } catch (err) {
      performance.now = realNow;
      realNowRef = null;
      return { ok: false, error: String(err) };
    }

    run = newRun();
    run.startedAt = realNow();
    state = 'running';
    result = null;

    let status = null;
    const liveStart = realNow();
    // Темп показа: 1.0 — обычная скорость игры (60 кадров в секунду),
    // 2.0 — вдвое быстрее. Физика в любом случае шагает ровно 1/60 c,
    // поэтому темп не влияет на поведение агента и на итоговый счёт.
    const speed = CFG.liveSpeed > 0 ? CFG.liveSpeed : 1;
    // Точка отсчёта темпа. Может сдвигаться, если окно подвисло: тогда игру
    // не проматываем рывком, а продолжаем с обычной скоростью.
    let paceStart = liveStart;

    function step() {
      if (state !== 'running') { return; }
      liveRaf = 0;

      // Сколько кадров должно быть проиграно к этому моменту реального
      // времени: за 1000 мс ровно 60 кадров — как в обычной игре.
      // Тиков может быть больше, чем кадров (монитор 144 Гц), поэтому при
      // budget = 0 просто ничего не двигаем. Раньше здесь стоял принудительный
      // минимум в 1 кадр — из-за него бег ускорялся в 2.4 раза.
      const elapsed = (realNow() - paceStart) * speed;
      let budget = Math.floor(elapsed / FRAME_MS) - run.frames;
      if (budget < 0) { budget = 0; }
      // Окно перекрыли или система усыпила таймеры — долг вырос больше чем на
      // секунду игры. Не проматываем её рывком: сдвигаем отсчёт, и дальше бег
      // идёт ровно с обычной скоростью, как в настоящей игре.
      if (budget > 60) {
        paceStart = realNow() - (run.frames * FRAME_MS) / speed;
        budget = 0;
      }

      for (let s = 0; s < budget && !status; s++) {
        if (runner.crashed) {
          status = 'DEAD';
          break;
        }
        const outputs = decide(runner);
        apply(runner, outputs[0], outputs[1]);
        clock += FRAME_MS;
        // Игра останавливает себя при потере фокуса окна (blur/visibility).
        // Для показа окно может уйти на второй план, поэтому держим её живой.
        if (!runner.playing) { runner.playing = true; runner.paused = false; }
        runner.update();
        trackJump(runner);
        run.frames += 1;
        if (runner.tRex && runner.tRex.ducking) { run.ducks += 1; }
        if (CFG.maxFrames > 0 && run.frames >= CFG.maxFrames) {
          status = 'FRAMES';
        } else if (CFG.seconds > 0 && run.frames * FRAME_MS / 1000 >= CFG.seconds) {
          status = 'TIME';
        } else if (run.frames >= 260000) {
          status = 'FRAMES';
        }
      }
      updateLiveOverlay(runner, runner.crashed ? 'GAME OVER' : 'RUNNING');

      if (runner.crashed) { status = 'DEAD'; }
      if (status) {
        trackStillPending();
        performance.now = realNowRef;
        realNowRef = null;
        state = 'done';
        result = snapshotResult(status, runner);
        result.seconds = Math.round(liveElapsed() / 10) / 100;
        updateLiveOverlay(runner, status === 'DEAD' ? 'GAME OVER' : status);
        try { runner.stop(); } catch (err) { /* noop */ }
        return;
      }
      scheduleLive(step);
    }

    function liveElapsed() {
      return run.frames * FRAME_MS;
    }

    function trackStillPending() {
      // Прыжок остался в воздухе и не накрыл препятствие — считаем бесполезным,
      // ровно так же, как это делает finish().
      if (run && run.jumpPending && !run.jumpSafe) {
        run.jumpPending = false;
        run.useless += 1;
        return true;
      }
      if (run) { run.jumpPending = false; }
      return false;
    }

    scheduleLive(step);
    return { ok: true };
  }

  function start() {
    return startLive();
  }

  function poll() {
    const runner = window.Runner && window.Runner.instance_;
    return {
      state: state,
      result: result,
      // Текущий прогресс — нужен консоли показа, чтобы было видно, как
      // растёт счёт, не глядя в окно.
      score: runner ? Math.round((runner.distanceRan || 0) * 0.025) : 0,
      frames: run ? run.frames : 0
    };
  }

  function abort(reason) {
    finish(reason || 'ABORT');
    return { state: state, result: result };
  }

  function info() {
    const runner = window.Runner && window.Runner.instance_;
    return {
      ready: !!(runner && window.Runner.imageSprite),
      playing: !!(runner && runner.playing),
      crashed: !!(runner && runner.crashed)
    };
  }

  window.__dino = {
    init: init,
    start: start,
    poll: poll,
    abort: abort,
    info: info,
    simulate: simulate
  };
})();
"""


# ---------------------------------------------------------------------------
# Перенос фенотипа NEAT в JavaScript
# ---------------------------------------------------------------------------
def network_to_js(genome: neat.DefaultGenome, config: neat.Config) -> dict[str, Any]:
    """Сериализует выраженную сеть генома в формат, понятный AGENT_JS."""
    genome_config = config.genome_config
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers, required = feed_forward_layers(
        genome_config.input_keys, genome_config.output_keys, connections
    )
    required_with_inputs = required.union(set(genome_config.input_keys))

    evals: list[dict[str, Any]] = []
    for layer in layers:
        for node in layer:
            links = []
            for conn_key in connections:
                inode, onode = conn_key
                if onode == node and inode in required_with_inputs:
                    links.append([inode, float(genome.connections[conn_key].weight)])
            node_gene = genome.nodes[node]
            evals.append(
                {
                    "node": node,
                    "act": node_gene.activation,
                    "agg": node_gene.aggregation,
                    "bias": float(node_gene.bias),
                    "response": float(node_gene.response),
                    "links": links,
                }
            )

    return {
        "inputs": list(genome_config.input_keys),
        "outputs": list(genome_config.output_keys),
        "evals": evals,
    }


# ---------------------------------------------------------------------------
# Результат эпизода
# ---------------------------------------------------------------------------
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
    trace: list[dict[str, Any]] = field(default_factory=list)


# Штраф за прыжок, сделанный заведомо раньше нужного. Без него у NEAT есть
# тривиальный локальный оптимум «прыгать непрерывно»: это даёт score 60-90,
# после чего эволюция перестаёт искать настоящее поведение.
USELESS_JUMP_PENALTY = 0.6

# Бонус за дожитие до лимита эпизода. Должен быть МАЛ по сравнению со счётом,
# иначе сеть учится «дожить до таймера», а не играть: при лимите 30 c выживший
# без умения получал 250 + 0.25*score, что перекрывало разницу в игре.
SURVIVAL_BONUS = 50.0


def make_result(data: dict[str, Any]) -> EpisodeResult:
    """Считает фитнес из сырых данных эпизода, полученных из страницы."""
    score = int(data.get("score", 0) or 0)
    frames = int(data.get("frames", 0) or 0)
    seconds = float(data.get("seconds", 0) or 0)
    status = str(data.get("status", "UNKNOWN"))
    jumps = int(data.get("jumps", 0) or 0)
    useless = int(data.get("useless", 0) or 0)
    ducks = int(data.get("ducks", 0) or 0)

    useful_share = 1.0 if jumps == 0 else max(0.0, 1.0 - useless / max(1, jumps))
    fitness = float(score) * (0.4 + 0.6 * useful_share) + 0.02 * frames
    fitness -= USELESS_JUMP_PENALTY * useless
    if status in ("TIME", "FRAMES"):
        # Дожил до лимита эпизода: небольшой бонус за выживание. Без второго
        # слагаемого вида +k*score: score уже учтён выше, дублирование
        # перекосило бы отбор в пользу «долгих, но пассивных» геномов.
        fitness += SURVIVAL_BONUS
    elif status in ("NO_RUNNER", "STALL", "ERROR", "ABORT", "PY_TIMEOUT"):
        fitness = -100.0

    return EpisodeResult(
        fitness=round(fitness, 3),
        score=score,
        frames=frames,
        seconds=seconds,
        status=status,
        jumps=jumps,
        useless=useless,
        ducks=ducks,
        trace=list(data.get("trace") or []),
    )


# ---------------------------------------------------------------------------
# Один браузер = один воркер
# ---------------------------------------------------------------------------
class BrowserWorker:
    """Отдельный браузер Chromium со своей страницей и своим агентом."""

    def __init__(
        self,
        index: int,
        playwright: Playwright,
        config: neat.Config,
        args: argparse.Namespace,
    ) -> None:
        self.index = index
        self.playwright = playwright
        self.config = config
        self.args = args
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.net: Optional[dict[str, Any]] = None
        # Каждый воркер получает свой сид трассы — так разные браузеры показывают
        # разные наборы препятствий, популяция обобщается лучше и не переобучается
        # под единственную трассу поколения.
        self.route_seed = getattr(args, "route_seed", None)
        self.agent_cfg = {
            "canvasWidth": CANVAS_W,
            "seconds": args.episode_seconds,
            "maxFrames": args.max_frames,
            "clearMs": args.clear_ms,
            "seed": self.route_seed if self.route_seed is not None else (1 + index),
            "trace": getattr(args, "trace", 0),
            "render": getattr(args, "render", False),
            "liveSpeed": getattr(args, "live_speed", 1.0),
            "pteroReflex": getattr(args, "ptero_reflex", True),
        }

    async def start(self) -> None:
        launch_args = [
            "--mute-audio",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            # Windows считает окно «перекрытым», если его закрыло другое окно,
            # и тогда Chromium замедляет таймеры страницы. Для показа это
            # недопустимо: бег должен идти в реальном темпе, даже если окно
            # на секунду оказалось позади другого.
            "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
        ]
        self.browser = await self.playwright.chromium.launch(
            headless=self.args.headless, args=launch_args
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=1,
        )
        self.page = await self.context.new_page()
        await self.page.add_init_script(script=AGENT_JS)
        await self._open_game()
        if not self.args.headless:
            # Выводим окно на передний план: при 15 одновременных браузерах
            # Chromium иначе открывает их позади терминала/редактора.
            try:
                await self.page.bring_to_front()
            except Exception:  # noqa: BLE001 — не критично для обучения
                pass

    async def _open_game(self) -> None:
        assert self.page is not None
        await self.page.goto(GAME_URL, wait_until="load", timeout=45_000)
        await self.page.wait_for_function(
            "() => !!(window.Runner && window.Runner.instance_ && window.Runner.imageSprite)",
            timeout=45_000,
        )
        await self.page.wait_for_function(
            "() => !!(window.__dino && window.__dino.info().ready)", timeout=15_000
        )

    async def _recover(self) -> None:
        """Перезагружает страницу после сбоя браузера/скрипта.

        Метод НИКОГДА не выбрасывает исключение: он вызывается из retry-цикла
        run_genome, и любая ошибка восстановления не должна ронять воркер.
        """
        try:
            if self.page is not None:
                await self._open_game()
                if self.net is not None:
                    await self.page.evaluate(
                        "(p) => window.__dino.init(p)",
                        {"net": self.net, "cfg": self.agent_cfg},
                    )
                return
        except Exception:  # noqa: BLE001 — падаем на жёсткий перезапуск
            pass

        # Мягкий путь не сработал (или страницы нет) — поднимаем браузер заново.
        try:
            if self.browser is not None:
                await self.browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await self.start()
            if self.net is not None and self.page is not None:
                await self.page.evaluate(
                    "(p) => window.__dino.init(p)",
                    {"net": self.net, "cfg": self.agent_cfg},
                )
        except Exception:  # noqa: BLE001
            # Совсем не поднялось — оставляем как есть, следующий retry попробует.
            pass

    async def run_genome(self, genome: neat.DefaultGenome) -> EpisodeResult:
        """Прогоняет один геном.

        По умолчанию используется быстрый детерминированный режим: страница
        прогоняет весь эпизод в одном вызове ``__dino.simulate()`` с
        замороженными часами и шагом 1/60 c. Это на порядки быстрее реального
        времени и даёт воспроизводимый результат.
        """
        assert self.page is not None
        self.net = network_to_js(genome, self.config)
        if getattr(self.args, "live", False):
            return await self._run_genome_live()

        hard_timeout = max(
            30.0,
            self.args.episode_seconds * 1.5 + 15.0,
            # Симуляция идёт ~50-150 тыс. кадров/с на браузер, но на 15
            # параллельных воркерах реальная скорость падает. Даём щедрый
            # запас по кадрам, чтобы длинный эпизод не считался таймаутом.
            self.args.max_frames / 1500.0 + 20.0,
        )
        total_timeout = hard_timeout + 20.0  # Additional buffer for browser overhead
        last_error = "unknown"

        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    await self._recover()
                await self.page.evaluate(
                    "(p) => window.__dino.init(p)",
                    {"net": self.net, "cfg": self.agent_cfg},
                )
                snapshot = await asyncio.wait_for(
                    self.page.evaluate("() => window.__dino.simulate()"),
                    timeout=total_timeout,
                )
                if not snapshot.get("state") == "done":
                    await asyncio.sleep(0.3)
                    continue
                return make_result(snapshot.get("result") or {})
            except asyncio.TimeoutError:
                last_error = "browser timeout"
                # Браузер может быть зависший — перезапускаем его полностью
                await self._force_restart()
                await asyncio.sleep(0.5)
                if attempt == 3:
                    return EpisodeResult(
                        fitness=-100.0,
                        score=0,
                        frames=0,
                        seconds=0.0,
                        status="PY_TIMEOUT",
                        error=last_error,
                    )
            except Exception as error:  # noqa: BLE001 — воркер не должен падать
                last_error = str(error)
                await asyncio.sleep(0.5)
                if attempt == 3:
                    return EpisodeResult(
                        fitness=-100.0,
                        score=0,
                        frames=0,
                        seconds=0.0,
                        status="ERROR",
                        error=last_error,
                    )

        return EpisodeResult(-100.0, 0, 0, 0.0, "ERROR", "retries exhausted")

    async def _force_restart(self) -> None:
        """Полный перезапуск браузера после зависания. Не выбрасывает исключений."""
        try:
            if self.browser is not None:
                await self.browser.close()
        except Exception:  # noqa: BLE001
            pass
        self.browser = None
        self.context = None
        self.page = None
        try:
            await self.start()
            if self.net is not None and self.page is not None:
                await self.page.evaluate(
                    "(p) => window.__dino.init(p)",
                    {"net": self.net, "cfg": self.agent_cfg},
                )
        except Exception:  # noqa: BLE001
            # Не поднялось — следующий retry вызовет _recover().
            pass

    async def _run_genome_live(self) -> EpisodeResult:
        """Живой режим (для --eval): игра идёт в реальном времени, с анимацией."""
        assert self.page is not None
        # Показ идёт в реальном времени: при обычной скорости 1.0 игровая
        # секунда занимает реальную секунду, поэтому таймаут считается с учётом
        # темпа и лимита кадров, а не только по длительности эпизода.
        speed = getattr(self.args, "live_speed", 1.0) or 1.0
        frames_seconds = (
            self.args.max_frames / 60.0 if self.args.max_frames > 0 else 0.0
        )
        planned = max(self.args.episode_seconds, frames_seconds) / speed
        # Если окно подвиснет, игра не проматывается рывком, а продолжается
        # с обычной скоростью — значит реальное время может оказаться больше
        # расчётного. Даём двойной запас, чтобы длинный показ не обрывался.
        hard_timeout = planned * 2.0 + 120.0
        last_error = "unknown"
        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    await self._recover()
                await self.page.evaluate(
                    "(p) => window.__dino.init(p)",
                    {"net": self.net, "cfg": self.agent_cfg},
                )
                started = await self.page.evaluate("() => window.__dino.start()")
                if not started.get("ok"):
                    await asyncio.sleep(0.3)
                    continue

                deadline = time.monotonic() + hard_timeout
                next_report = time.monotonic() + 10.0
                while True:
                    await asyncio.sleep(self.args.poll_interval)
                    snapshot = await self.page.evaluate("() => window.__dino.poll()")
                    if snapshot.get("state") == "done":
                        return make_result(snapshot.get("result") or {})
                    # Раз в 10 секунд печатаем текущий счёт: при обычной
                    # скорости показ длится минутами, и так видно прогресс.
                    now = time.monotonic()
                    if now >= next_report:
                        next_report = now + 10.0
                        score = snapshot.get("score")
                        frames = snapshot.get("frames")
                        if score is not None:
                            print(
                                f"  счёт {score} (кадр {frames})",
                                flush=True,
                            )
                    if now > deadline:
                        await self.page.evaluate(
                            "() => window.__dino.abort('PY_TIMEOUT')"
                        )
                        return make_result({"status": "PY_TIMEOUT"})
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                await asyncio.sleep(0.5)
                if attempt == 3:
                    return EpisodeResult(-100.0, 0, 0, 0.0, "ERROR", last_error)
        return EpisodeResult(-100.0, 0, 0, 0.0, "ERROR", "retries exhausted")

    async def close(self) -> None:
        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Конкурентная оценка популяции на пуле браузеров
# ---------------------------------------------------------------------------
class AsyncEvaluator:
    """Раздаёт геномы воркерам; 15 браузеров играют одновременно."""

    def __init__(
        self,
        playwright: Playwright,
        config: neat.Config,
        args: argparse.Namespace,
    ) -> None:
        self.playwright = playwright
        self.config = config
        self.args = args
        self.workers: list[BrowserWorker] = []
        self.results: dict[int, EpisodeResult] = {}

    async def __aenter__(self) -> "AsyncEvaluator":
        self.workers = [
            BrowserWorker(index, self.playwright, self.config, self.args)
            for index in range(self.args.browsers)
        ]
        await asyncio.gather(*(worker.start() for worker in self.workers))
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers))

    def set_route_seed(self, seed: int) -> None:
        """Сиды трассы на поколение: базовый + индекс воркера.

        Все браузеры поколения получают близкие, но разные сиды — так геномы
        оцениваются на 다양ных трассах, не переобучаясь под одну.
        """
        for worker in self.workers:
            worker.agent_cfg["seed"] = seed + worker.index

    async def evaluate(self, genomes: list[tuple[int, neat.DefaultGenome]]) -> dict[int, EpisodeResult]:
        queue: asyncio.Queue[Optional[tuple[int, neat.DefaultGenome]]] = asyncio.Queue()
        for item in genomes:
            queue.put_nowait(item)
        for _ in self.workers:
            queue.put_nowait(None)

        results: dict[int, EpisodeResult] = {}
        completed = 0
        total = len(genomes)
        lock = asyncio.Lock()

        async def consume(worker: BrowserWorker) -> None:
            nonlocal completed
            while True:
                item = await queue.get()
                if item is None:
                    break
                genome_id, genome = item
                outcomes = []
                # Один удачный seed не должен делать геном чемпионом. При
                # robustness-оценке геном проходит несколько независимых трасс,
                # а отбор ориентируется на худший результат.
                repeats = max(1, int(getattr(self.args, "fitness_seeds", 1)))
                base_seed = self.args.route_seed
                for repeat in range(repeats):
                    if base_seed is not None:
                        worker.agent_cfg["seed"] = (
                            base_seed + worker.index + repeat * 1_000_003
                        )
                    outcomes.append(await worker.run_genome(genome))
                outcome = min(outcomes, key=lambda value: (value.score, value.fitness))
                genome.fitness = outcome.fitness
                results[genome_id] = outcome
                async with lock:
                    completed += 1
                    if completed % max(1, total // 10) == 0 or completed == total:
                        print(
                            f"    оценено {completed}/{total}; "
                            f"счёт={outcome.score} прыжков={outcome.jumps} "
                            f"(лишних {outcome.useless})",
                            flush=True,
                        )

        await asyncio.gather(*(consume(worker) for worker in self.workers))
        return results


# ---------------------------------------------------------------------------
# Журнал поколений
# ---------------------------------------------------------------------------
class TrainingLog:
    HEADERS = [
        "generation", "best_fitness", "mean_fitness", "best_score",
        "min_score", "max_score", "mean_score", "errors", "workers", "seconds",
    ]

    def __init__(self, path: str) -> None:
        self.path = path
        need_header = not os.path.exists(path)
        if os.path.exists(path):
            # Если в старом логе меньше колонок, чем сейчас — перезаписываем с новыми.
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    first_line = handle.readline()
                existing = [h.strip() for h in first_line.split(",")]
                if existing != self.HEADERS:
                    need_header = True
            except Exception:  # noqa: BLE001
                need_header = True
        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.HEADERS)
        if need_header:
            self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({key: row.get(key, "") for key in self.HEADERS})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def save_best(genome: neat.DefaultGenome, config: neat.Config, args: argparse.Namespace) -> None:
    with open(args.model_file, "wb") as handle:
        pickle.dump(genome, handle)
    payload = network_to_js(genome, config)
    with open(args.network_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"Лучший геном сохранён: {args.model_file} и {args.network_file}")


def save_elite(
    genomes: list[tuple[neat.DefaultGenome, EpisodeResult]],
    config: neat.Config,
    args: argparse.Namespace,
    top_n: int = 5,
) -> None:
    """Сохраняет топ-N геномов в папку elite/ для отбора и анализа."""
    elite_dir = os.path.join(os.path.dirname(args.model_file) or ".", "elite")
    os.makedirs(elite_dir, exist_ok=True)
    ranked = sorted(genomes, key=lambda item: item[1].fitness or -1e9, reverse=True)
    for rank, (genome, result) in enumerate(ranked[:top_n], start=1):
        prefix = f"elite-{rank:02d}_score{result.score}_fit{result.fitness:.0f}"
        with open(os.path.join(elite_dir, prefix + ".pkl"), "wb") as handle:
            pickle.dump(genome, handle)
        payload = network_to_js(genome, config)
        with open(os.path.join(elite_dir, prefix + ".json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Элита ({min(top_n, len(ranked))} геномов) сохранена в {elite_dir}/")


def load_genome(path: str) -> neat.DefaultGenome:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def seed_population(
    population: neat.Population, config: neat.Config, path: str
) -> None:
    """Засевает популяцию клонами обученного генома с мутациями.

    Нужно после изменения среды (например, когда добавлен присед под птиц):
    случайная популяция заново учится играть с нуля, а клоны уже умеют бегать
    и доучиваются только новому поведению.
    """
    base = load_genome(path)
    seeders: dict[int, neat.DefaultGenome] = {}
    for genome_id in range(1, config.pop_size + 1):
        clone = neat.DefaultGenome(genome_id)
        clone.configure_new(config.genome_config)
        clone.nodes = {k: v.copy() for k, v in base.nodes.items()}
        clone.connections = {k: v.copy() for k, v in base.connections.items()}
        clone.fitness = None
        if genome_id > 1:
            # Мутации дают популяции разнообразие вокруг рабочей точки.
            clone.mutate(config.genome_config)
        seeders[genome_id] = clone

    population.population = seeders
    population.species.speciate(config, population.population, population.generation)
    print(
        f"Популяция засеяна {len(seeders)} клонами генома {path} "
        f"(первый — без мутаций, остальные — с мутациями)."
    )


async def train(args: argparse.Namespace, config: neat.Config) -> None:
    config.pop_size = args.population
    config.no_fitness_termination = True

    if args.resume:
        print(f"Восстанавливаем состояние из {args.resume} ...")
        population = neat.Checkpointer.restore_checkpoint(args.resume, new_config=config)
    else:
        population = neat.Population(config)
        # Засев: если есть обученный геном, кладём его клоны с мутациями вместо
        # случайных сетей. Так дообучение идёт не «с нуля», а от уже рабочей
        # игры — важно, когда в среду добавили новое поведение (присед).
        if args.seed_model:
            seed_population(population, config, args.seed_model)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(args.checkpoint_every, filename_prefix=args.checkpoint_prefix)
    )

    log = TrainingLog(args.log_file)
    best_ever: Optional[neat.DefaultGenome] = None
    best_score_ever = 0
    generation_history: list[dict[str, Any]] = []
    # Запоминаем стартовую длину эпизода: curriculum должен интерполировать
    # от неё к максимуму, а не накапливать приращения в args.episode_seconds.
    curriculum_start_seconds = args.episode_seconds

    # Не теряем более удачный геном из прошлых запусков.
    if os.path.exists(args.model_file):
        try:
            previous = load_genome(args.model_file)
            if previous.fitness is not None:
                best_ever = previous
                print(f"Учтён сохранённый геном: fitness={previous.fitness:.1f}")
        except Exception as error:  # noqa: BLE001
            print(f"Старый геном не загрузился ({error}) — начнём учёт заново.")

    print(
        f"Старт: браузеров={args.browsers}, поколений={args.generations}, "
        f"популяция={config.pop_size}, эпизод={args.episode_seconds} c"
    )
    if args.curriculum:
        print(
            f"Curriculum включён: длина эпизода растёт от {args.episode_seconds} "
            f"до {args.curriculum_max_seconds} с."
        )
    if args.headless:
        print(
            "Окна браузеров скрыты (headless) — так обучение идёт быстрее. "
            "Чтобы видеть игру, добавьте --headed."
        )

    try:
        async with async_playwright() as playwright:
            async with AsyncEvaluator(playwright, config, args) as evaluator:
                for _ in range(args.generations):
                    started_at = time.monotonic()
                    generation = population.generation
                    print(f"\n=== Поколение {generation} ===", flush=True)
                    population.reporters.start_generation(generation)

                    # --- Curriculum: постепенно усложняем окружение ---
                    if args.curriculum:
                        progress = min(1.0, generation / max(1, args.generations - 1))
                        args.episode_seconds = curriculum_start_seconds + progress * (
                            args.curriculum_max_seconds - curriculum_start_seconds
                        )
                        args.max_frames = max(
                            args.max_frames,
                            int(args.episode_seconds * 60) + 1,
                        )
                        for worker in evaluator.workers:
                            worker.agent_cfg["seconds"] = args.episode_seconds
                            worker.agent_cfg["maxFrames"] = args.max_frames
                        print(f"  длина эпизода: {args.episode_seconds:.1f} c", flush=True)

                    # У всех геномов поколения — разные трассы (по одному сиду на воркер),
                    # чтобы сеть обобщалась, не переобучаясь под одну карту.
                    if args.route_seed is None:
                        route_seed = random.randrange(1, 2**31 - 1)
                    else:
                        route_seed = args.route_seed + generation
                    evaluator.set_route_seed(route_seed)
                    print(f"  сид трассы: {route_seed}", flush=True)

                    genomes = list(population.population.items())
                    results = await evaluator.evaluate(genomes)

                    scored = [(population.population[gid], res) for gid, res in results.items()]
                    best_genome, best_result = max(scored, key=lambda item: item[0].fitness or -1e9)
                    mean_fitness = sum(g.fitness or 0.0 for g, _ in scored) / max(1, len(scored))
                    scores = [res.score for _, res in scored]
                    errors = sum(
                        1
                        for _, res in scored
                        if res.status in ("ERROR", "NO_RUNNER", "STALL", "ABORT", "PY_TIMEOUT")
                    )
                    n_workers = len(evaluator.workers)
                    elapsed = time.monotonic() - started_at

                    # Сохраняем по реальному счёту: fitness с бонусом TIME может
                    # завышать «дожившего до лимита», но не умеющего играть геном.
                    improved = best_result.score > best_score_ever or (
                        best_result.score == best_score_ever
                        and (best_genome.fitness or 0) > (best_ever.fitness if best_ever else -1e9)
                    )
                    if best_ever is None or improved:
                        best_ever = best_genome
                        best_score_ever = best_result.score
                        save_best(best_genome, config, args)

                    # Элита топ-N каждые N поколений
                    if args.elite_every > 0 and generation % args.elite_every == 0:
                        save_elite(scored, config, args, top_n=5)

                    print(
                        f"  лучший score={best_result.score}, fitness={best_genome.fitness:.1f} "
                        f"({best_result.status}), средний fitness={mean_fitness:.1f}, "
                        f"ошибок={errors}, {elapsed:.1f} c"
                    )
                    row = {
                        "generation": generation,
                        "best_fitness": round(best_genome.fitness or 0.0, 2),
                        "mean_fitness": round(mean_fitness, 2),
                        "best_score": best_result.score,
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "mean_score": round(sum(scores) / max(1, len(scores)), 2),
                        "errors": errors,
                        "workers": n_workers,
                        "seconds": round(elapsed, 1),
                    }
                    log.write(row)
                    generation_history.append(row)

                    population.reporters.post_evaluate(
                        config, population.population, population.species, best_genome
                    )

                    population.population = population.reproduction.reproduce(
                        config, population.species, config.pop_size, generation
                    )
                    if not population.species.species:
                        population.reporters.complete_extinction()
                        population.population = population.reproduction.create_new(
                            config.genome_type, config.genome_config, config.pop_size
                        )
                    population.species.speciate(config, population.population, generation)
                    population.reporters.end_generation(
                        config, population.population, population.species
                    )
                    population.generation += 1

    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    finally:
        log.close()

    print(f"\nОбучение завершено. Лучший счёт за всё время: {best_score_ever}")
    if best_ever is not None:
        save_best(best_ever, config, args)

    # --- Дашборд ---
    if args.dashboard and generation_history:
        generate_dashboard(args.log_file, "training_dashboard.html")
        print(f"Дашборд сохранён: training_dashboard.html")


def generate_dashboard(log_path: str, output_path: str) -> None:
    """Генерирует HTML-дашборд с графиками обучения из CSV-лога."""
    try:
        import csv

        generations = []
        best_fitness = []
        mean_fitness = []
        best_score = []
        mean_score = []
        errors = []

        with open(log_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    generations.append(int(row["generation"]))
                    best_fitness.append(float(row["best_fitness"]))
                    mean_fitness.append(float(row["mean_fitness"]))
                    best_score.append(int(row["best_score"]))
                    mean_score.append(float(row["mean_score"]))
                    errors.append(int(row["errors"]))
                except (ValueError, KeyError):
                    continue

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Dino NEAT — Дашборд обучения</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
  canvas {{ width: 100%; height: 200px; }}
  .stat {{ font-size: 1.25rem; font-weight: bold; color: #0066cc; }}
  .muted {{ color: #666; font-size: 0.9rem; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>🦖 Dino NEAT — Дашборд обучения</h1>
<p class="muted">Источник: {log_path} | поколений: {len(generations)}</p>
<div class="grid">
  <div class="card">
    <h3>Лучший фитнес</h3>
    <canvas id="bestFit"></canvas>
  </div>
  <div class="card">
    <h3>Средний фитнес</h3>
    <canvas id="meanFit"></canvas>
  </div>
  <div class="card">
    <h3>Лучший счёт</h3>
    <canvas id="bestScore"></canvas>
  </div>
  <div class="card">
    <h3>Средний счёт</h3>
    <canvas id="meanScore"></canvas>
  </div>
  <div class="card">
    <h3>Ошибок воркеров</h3>
    <canvas id="errors"></canvas>
  </div>
</div>
<script>
const generations = {generations!r};
const best_fitness = {best_fitness!r};
const mean_fitness = {mean_fitness!r};
const best_score = {best_score!r};
const mean_score = {mean_score!r};
const errors = {errors!r};

const commonOpts = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{ y: {{ beginAtZero: true }}, x: {{ title: {{ display: true, text: 'Поколение' }} }} }},
}};
function chart(id, label, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{
      labels: generations,
      datasets: [{{
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: color + '22',
        tension: 0.3,
        fill: true,
      }}],
    }},
    options: commonOpts,
  }});
}}
chart('bestFit', 'Лучший фитнес', best_fitness, '#0066cc');
chart('meanFit', 'Средний фитнес', mean_fitness, '#666666');
chart('bestScore', 'Лучший счёт', best_score, '#00aa44');
chart('meanScore', 'Средний счёт', mean_score, '#aa6600');
chart('errors', 'Ошибок', errors, '#cc3333');
</script>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
    except Exception as error:  # noqa: BLE001
        print(f"Не удалось создать дашборд: {error}")


# ---------------------------------------------------------------------------
# Демонстрация лучшего генома
# ---------------------------------------------------------------------------
async def play(args: argparse.Namespace, config: neat.Config) -> None:
    if not os.path.exists(args.model_file):
        print(f"Не найден файл модели {args.model_file}. Сначала обучите сеть.")
        return

    genome = load_genome(args.model_file)
    print(f"Играем лучшим геномом: {args.model_file}")

    args.headless = False
    # Демонстрация должна показывать настоящую игру, а не статичный результат.
    # Live-режим оставляет штатный Runner canvas и его спрайты включёнными.
    args.live = True
    args.render = True
    args.episode_seconds = args.eval_seconds
    # Оставляем лимит, заданный командой. Для показа обычно используется
    # 70000 кадров, как и в независимой проверке evaluate.py.
    if args.max_frames <= 0:
        args.max_frames = 70_000
    if args.route_seed is None:
        args.route_seed = random.randrange(1, 2**31 - 1)

    print(
        f"Открываю окно Chromium. Бег пойдёт в обычном темпе игры, "
        f"покадрово (сид трассы {args.route_seed}).",
        flush=True,
    )

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, args)
        await worker.start()
        try:
            result = await worker.run_genome(genome)
            print(
                f"Результат: статус={result.status}, счёт={result.score}, "
                f"кадров={result.frames}, время={result.seconds:.1f} c",
                flush=True,
            )
            print(
                "Игра окончена. Окно остаётся открытым — закройте его вручную.",
                flush=True,
            )

            # Ждём закрытия окна, а не фиксированный таймаут: учитель сам решает,
            # сколько смотреть на итоговый счёт. Ctrl+C в терминале тоже работает.
            closed = asyncio.get_running_loop().create_future()

            def on_disconnected(*_args: object) -> None:
                if not closed.done():
                    closed.set_result(True)

            worker.browser.on("disconnected", on_disconnected)
            if worker.browser.is_connected():
                try:
                    await closed
                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass
        finally:
            await worker.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dino Runner — обучение сети NEAT в пуле браузеров Chromium"
    )
    parser.add_argument("--generations", type=int, default=30, help="число поколений")
    parser.add_argument("--browsers", type=int, default=15, help="сколько браузеров запускать")
    parser.add_argument("--population", type=int, default=150, help="размер популяции NEAT")
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=600.0,
        help="лимит эпизода в секундах (0 — без лимита по времени)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=80000,
        help="лимит кадров эпизода (0 — без лимита; страховка от вечного цикла)",
    )
    parser.add_argument("--clear-ms", type=int, default=900, help="пауза до первых препятствий")
    parser.add_argument(
        "--route-seed",
        type=int,
        default=None,
        help="сид трассы; по умолчанию случайный на поколение (одинаков для всей популяции)",
    )
    parser.add_argument(
        "--fitness-seeds",
        type=int,
        default=1,
        help="число независимых трасс для fitness каждого генома",
    )
    parser.add_argument("--poll-interval", type=float, default=0.15, help="опрос страницы, c")
    parser.add_argument("--config", default=NEAT_CONFIG, help="файл конфигурации NEAT")
    parser.add_argument("--resume", default=None, help="файл чекпоинта для продолжения")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="чекпоинт каждые N поколений")
    parser.add_argument(
        "--checkpoint-prefix",
        default="neat-checkpoint-",
        help="префикс файлов чекпоинтов",
    )
    parser.add_argument(
        "--seed-model",
        default=None,
        help="геном, клонами которого засеять популяцию перед обучением",
    )
    parser.add_argument("--model-file", default="best_genome.pkl", help="файл лучшего генома")
    parser.add_argument("--network-file", default="best_network.json", help="JSON-описание сети")
    parser.add_argument("--log-file", default="training_log.csv", help="CSV-журнал")
    parser.add_argument("--headed", action="store_true", help="показывать окна браузеров")
    parser.add_argument(
        "--eval-seconds",
        type=float,
        default=120.0,
        help="длительность показа в режиме --eval",
    )
    parser.add_argument("--eval", action="store_true", help="только играть сохранённым геномом")
    parser.add_argument(
        "--live-speed",
        type=float,
        default=1.0,
        help="темп показа в режиме --eval: 1 — обычная скорость игры, 2 — вдвое быстрее",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="постепенно увеличивать длину эпизода с поколения к поколению",
    )
    parser.add_argument(
        "--curriculum-max-seconds",
        type=float,
        default=1000.0,
        help="максимальная длина эпизода в режиме curriculum",
    )
    parser.add_argument(
        "--elite-every",
        type=int,
        default=10,
        help="сохранять элиту топ-N геномов каждые N поколений (0 — отключить)",
    )
    parser.add_argument(
        "--trace",
        type=int,
        default=0,
        help="сохранять N последних кадров эпизода для диагностики падений",
    )
    parser.add_argument(
        "--no-ptero-reflex",
        dest="ptero_reflex",
        action="store_false",
        help="отключить детерминированный присед под птеродактилями",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="после обучения сгенерировать HTML-дашборд с графиками",
    )
    args = parser.parse_args()
    args.headless = not args.headed
    return args


def build_config(path: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(f"Не найден конфиг NEAT: {args.config}")
    config = build_config(args.config)
    if args.eval:
        asyncio.run(play(args, config))
    else:
        asyncio.run(train(args, config))


if __name__ == "__main__":
    main()
