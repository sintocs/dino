"""Как агент реагирует на птеродактилей разной высоты.

Считает по кадрам, где впереди птица: какой манёвр выбирает агент
(прыжок/присед/бег) в зависимости от высоты птицы, и переживает ли встречу.
"""

import argparse
import asyncio
from collections import Counter, defaultdict

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(seed: int, frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=True, episode_seconds=1000.0,
        max_frames=frames, clear_ms=900, poll_interval=0.05,
        route_seed=seed, live=False, trace=200000, render=False, live_speed=1.0,
    )


async def main() -> None:
    seeds = [777, 12345, 2026, 424242, 9001]
    frames = 60000
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesState if hasattr(neat, "DefaultSpeciesState")
        else neat.DefaultSpeciesSet,
        neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        for seed in seeds:
            worker = BrowserWorker(0, playwright, config, make_args(seed, frames))
            await worker.start()
            try:
                result = await worker.run_genome(genome)
                trace = getattr(result, "trace", None)
                if not trace:
                    snap = await worker.page.evaluate(
                        "() => window.__dino.poll()"
                    )
                    trace = (snap.get("result") or {}).get("trace") or []
                print(f"\n=== seed {seed}: score={result.score} "
                      f"frames={result.frames} status={result.status} "
                      f"trace={len(trace)} ===")
                analyze(trace)
            finally:
                await worker.close()


def analyze(trace: list) -> None:
    """Сводит кадры в «встречи» с птицами и смотрит выбор агента."""
    # Ищем первый кадр каждой встречи: птица впереди и ещё далеко.
    seen = set()
    reactions = defaultdict(Counter)
    for row in trace:
        if row.get("ot") != "PTER":
            continue
        oy = row.get("oy")
        dx = row.get("dx")
        if dx is None or dx < 60 or dx > 200:
            continue
        key = (row.get("f"),)
        if key in seen:
            continue
        seen.add(key)
        # Что делает агент, пока птица в зоне реакции.
        jump = row.get("jv") or 0
        duck = row.get("dv") or 0
        if jump > 0.5:
            act = "jump"
        elif duck > 0.5:
            act = "duck"
        else:
            act = "run"
        reactions[oy][act] += 1

    if not reactions:
        print("  птиц в зоне реакции не найдено")
        return
    for oy in sorted(reactions, key=lambda v: (v is None, v)):
        total = sum(reactions[oy].values())
        parts = ", ".join(f"{k}={v}" for k, v in reactions[oy].most_common())
        print(f"  yPos={oy}: всего {total} ({parts})")


if __name__ == "__main__":
    asyncio.run(main())
