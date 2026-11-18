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
    "is_cactus_large",
    "is_cactus_small",
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
    "blocks_ducking",   # помешает ли препятствие даже приседу
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
    render: false
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
    maxabs: xs => xs.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
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

  function buildInputs(runner) {
    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
    const w = CFG.canvasWidth || 600;
    const obstacle = obstacles.length ? obstacles[0] : null;
    const speed = runner.currentSpeed || 6;
    const f = new Array(19).fill(0);

    // Расстояние нормируем по рабочему окну реакции (~160 px), а не по всей
    // ширине канваса: иначе 400 и 120 px дают почти одинаковые 0.67 и 0.20.
    const dnorm = x => clamp(x / 160, -1, 3);

    if (!obstacle) {
      f[0] = 3; f[1] = 3; f[11] = 3;
    } else {
      const dx1 = obstacle.xPos - dino.xPos;
      f[0] = dnorm(dx1);
      f[1] = dnorm(obstacles.length > 1 ? (obstacles[1].xPos - dino.xPos) : w);
      f[2] = obstacles.length > 1 ? 1 : 0;

      const type = obstacle.typeConfig ? obstacle.typeConfig.type : '';
      const oh = obstacle.typeConfig ? obstacle.typeConfig.height : 0;
      const oy = obstacle.yPos || 0;
      const isPtero = type === 'PTERODACTYL';

      f[3] = isPtero ? 1 : 0;
      f[4] = isPtero ? clamp(oy / 150, -1, 1) : 0;
      f[5] = type === 'CACTUS_LARGE' ? 1 : 0;
      f[6] = type === 'CACTUS_SMALL' ? 1 : 0;
      f[7] = clamp(oh / 50, 0, 2);
      f[8] = clamp(obstacle.width / 50, 0, 2);
      f[9] = clamp(oy / 150, -1, 1);
      f[10] = clamp((oy + oh) / 150, -1, 1);
      f[11] = dnorm(dx1 - obstacle.width);
      // Перекрывает ли препятствие вертикаль приседа (низ дино при приседе = 118).
      f[18] = (oy + oh > 118 && oy < 93) ? 1 : 0;
    }

    const airborne = Math.max(0, dino.groundYPos - dino.yPos);
    const jumpVelocity = dino.jumpVelocity;
    f[12] = clamp(airborne / 50, 0, 1);
    f[13] = clamp((typeof jumpVelocity === 'number' ? jumpVelocity : 0) / 15, -2, 2);
    f[14] = dino.jumping ? 1 : 0;
    f[15] = dino.ducking ? 1 : 0;
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
        else { dino.setDuck(true); }
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

  function finish(status) {
    if (state !== 'running') { return; }
    if (liveRaf) { cancelAnimationFrame(liveRaf); liveRaf = 0; }
    const runner = window.Runner && window.Runner.instance_;
    if (run && run.jumpPending && !run.jumpSafe) {
      // Прыжок остался «в воздухе» и не накрыл препятствие.
      run.useless += 1;
    }
    if (run) { run.jumpPending = false; }
    if (realNowRef) { performance.now = realNowRef; realNowRef = null; }
    state = 'done';
    result = snapshotResult(status, runner);
    try { if (runner) { runner.stop(); } } catch (err) { /* noop */ }
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
    if (noRender) {
      runner.clearCanvas = function () {};
      runner.invert = function () {};
      const ctx = runner.canvasCtx;
      if (ctx && ctx.drawImage) { ctx.drawImage = function () {}; }
    }
    runner.restart();
  }

  function snapshotResult(status, runner) {
    return {
      status: status,
      score: Math.round((runner.distanceRan || 0) * 0.025),
      frames: run.frames,
      seconds: Math.round(run.frames * FRAME_MS / 10) / 100,
      crashed: !!runner.crashed,
      jumps: run.jumps,
      useless: run.useless,
      ducks: run.ducks
    };
  }

  function newRun() {
    return {
      frames: 0, jumps: 0, useless: 0, ducks: 0,
      jumpPending: false, jumpSafe: false
    };
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

    const maxByTime = Math.ceil((CFG.seconds || 30) * 60);
    const maxByFrames = CFG.maxFrames > 0 ? CFG.maxFrames : Infinity;
    const frameLimit = Math.min(maxByTime, maxByFrames);
    let status = 'FRAMES';

    try {
      for (let i = 0; i < frameLimit; i++) {
        if (runner.crashed) { status = 'DEAD'; break; }
        const outputs = activate(buildInputs(runner));
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
    if (liveRaf) { cancelAnimationFrame(liveRaf); liveRaf = 0; }

    installSeededRandom();

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

    let last = realNow();
    let acc = 0;
    let status = null;

    function step() {
      if (state !== 'running') { return; }
      liveRaf = 0;

      // Догоняем реальное время целыми кадрами, но не больше 4 за раз,
      // чтобы не впасть в «спираль смерти» после подвисания окна.
      const nowReal = realNow();
      acc += nowReal - last;
      last = nowReal;
      let steps = Math.floor(acc / FRAME_MS);
      if (steps > 4) { steps = 4; acc = 0; } else { acc -= steps * FRAME_MS; }

      for (let s = 0; s < steps; s++) {
        if (runner.crashed) { status = 'DEAD'; break; }
        const outputs = activate(buildInputs(runner));
        apply(runner, outputs[0], outputs[1]);
        clock += FRAME_MS;
        runner.update();
        trackJump(runner);
        run.frames += 1;
        if (runner.tRex && runner.tRex.ducking) { run.ducks += 1; }
        if (run.frames >= CFG.maxFrames) { status = 'FRAMES'; break; }
        if (run.frames * FRAME_MS / 1000 >= CFG.seconds) { status = 'TIME'; break; }
      }

      if (runner.crashed) { status = 'DEAD'; }
      if (status) {
        trackStillPending();
        performance.now = realNowRef;
        realNowRef = null;
        state = 'done';
        result = snapshotResult(status, runner);
        result.seconds = Math.round(liveElapsed() / 10) / 100;
        try { runner.stop(); } catch (err) { /* noop */ }
        return;
      }
      liveRaf = requestAnimationFrame(step);
    }

    function liveElapsed() {
      return run.frames * FRAME_MS;
    }

    function trackStillPending() {
      if (run && run.jumpPending && !run.jumpSafe) {
        run.jumpPending = false;
        return true;
      }
      return false;
    }

    liveRaf = requestAnimationFrame(step);
    return { ok: true };
  }

  function start() {
    return startLive();
  }

  function poll() {
    return { state: state, result: result };
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


# Штраф за прыжок, сделанный заведомо раньше нужного. Без него у NEAT есть
# тривиальный локальный оптимум «прыгать непрерывно»: это даёт score 60-90,
# после чего эволюция перестаёт искать настоящее поведение.
USELESS_JUMP_PENALTY = 0.6


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
    fitness = float(score) * (0.3 + 0.7 * useful_share) + 0.02 * frames
    fitness -= USELESS_JUMP_PENALTY * useless
    if status in ("TIME", "FRAMES"):
        # Дожил до лимита эпизода — это лучший возможный результат.
        fitness += 500.0 + 0.5 * score
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
        self.agent_cfg = {
            "canvasWidth": CANVAS_W,
            "seconds": args.episode_seconds,
            "maxFrames": args.max_frames,
            "clearMs": args.clear_ms,
            "seed": getattr(args, "route_seed", None) or 1,
        }

    async def start(self) -> None:
        launch_args = [
            "--mute-audio",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ]
        self.browser = await self.playwright.chromium.launch(
            headless=self.args.headless, args=launch_args
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1000, "height": 500},
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
        """Перезагружает страницу после сбоя браузера/скрипта."""
        try:
            if self.page is not None:
                await self._open_game()
                if self.net is not None:
                    await self.page.evaluate(
                        "(p) => window.__dino.init(p)",
                        {"net": self.net, "cfg": self.agent_cfg},
                    )
        except Exception:
            if self.browser is not None:
                try:
                    await self.browser.close()
                except Exception:
                    pass
            await self.start()
            if self.net is not None and self.page is not None:
                await self.page.evaluate(
                    "(p) => window.__dino.init(p)",
                    {"net": self.net, "cfg": self.agent_cfg},
                )

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
        hard_timeout = max(30.0, self.args.episode_seconds * 1.5 + 15.0)

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
                    timeout=hard_timeout,
                )
                if not snapshot.get("state") == "done":
                    await asyncio.sleep(0.3)
                    continue
                return make_result(snapshot.get("result") or {})
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

    async def _run_genome_live(self) -> EpisodeResult:
        """Живой режим (для --eval): игра идёт в реальном времени, с анимацией."""
        assert self.page is not None
        hard_timeout = self.args.episode_seconds + 20.0
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
                while True:
                    await asyncio.sleep(self.args.poll_interval)
                    snapshot = await self.page.evaluate("() => window.__dino.poll()")
                    if snapshot.get("state") == "done":
                        return make_result(snapshot.get("result") or {})
                    if time.monotonic() > deadline:
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
        """Один сид трассы на всё поколение — честное сравнение геномов."""
        for worker in self.workers:
            worker.agent_cfg["seed"] = seed

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
                outcome = await worker.run_genome(genome)
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
    HEADERS = ["generation", "best_fitness", "mean_fitness", "best_score", "seconds"]

    def __init__(self, path: str) -> None:
        self.path = path
        is_new = not os.path.exists(path)
        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.HEADERS)
        if is_new:
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


def load_genome(path: str) -> neat.DefaultGenome:
    with open(path, "rb") as handle:
        return pickle.load(handle)


async def train(args: argparse.Namespace, config: neat.Config) -> None:
    config.pop_size = args.population
    config.no_fitness_termination = True

    if args.resume:
        print(f"Восстанавливаем состояние из {args.resume} ...")
        population = neat.Checkpointer.restore_checkpoint(args.resume, new_config=config)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(args.checkpoint_every, filename_prefix="neat-checkpoint-")
    )

    log = TrainingLog(args.log_file)
    best_ever: Optional[neat.DefaultGenome] = None
    best_score_ever = 0

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

                    # У всех геномов поколения — одна и та же трасса, но от
                    # поколения к поколению она меняется, чтобы не переобучиться
                    # под единственный набор препятствий.
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

                    print(
                        f"  лучший score={best_result.score}, fitness={best_genome.fitness:.1f} "
                        f"({best_result.status}), средний fitness={mean_fitness:.1f}, {elapsed:.1f} c"
                    )
                    log.write(
                        {
                            "generation": generation,
                            "best_fitness": round(best_genome.fitness or 0.0, 2),
                            "mean_fitness": round(mean_fitness, 2),
                            "best_score": best_result.score,
                            "seconds": round(elapsed, 1),
                        }
                    )

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
    args.live = True
    args.episode_seconds = args.eval_seconds
    args.max_frames = 10_000_000
    if args.route_seed is None:
        args.route_seed = random.randrange(1, 2**31 - 1)

    print(
        f"Открываю окно Chromium на {args.eval_seconds:g} c "
        f"(сид трассы {args.route_seed}) ..."
    )

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, args)
        await worker.start()
        try:
            result = await worker.run_genome(genome)
            print(
                f"Результат: статус={result.status}, счёт={result.score}, "
                f"кадров={result.frames}, время={result.seconds:.1f} c"
            )
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
    parser.add_argument("--episode-seconds", type=float, default=30.0, help="лимит эпизода")
    parser.add_argument("--max-frames", type=int, default=4000, help="лимит кадров эпизода")
    parser.add_argument("--clear-ms", type=int, default=900, help="пауза до первых препятствий")
    parser.add_argument(
        "--route-seed",
        type=int,
        default=None,
        help="сид трассы; по умолчанию случайный на поколение (одинаков для всей популяции)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.15, help="опрос страницы, c")
    parser.add_argument("--config", default=NEAT_CONFIG, help="файл конфигурации NEAT")
    parser.add_argument("--resume", default=None, help="файл чекпоинта для продолжения")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="чекпоинт каждые N поколений")
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
