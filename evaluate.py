"""Валидация сохранённого генома на нескольких независимых трассах.

Запускает детерминированную симуляцию (без анимации) для одного и того же
генома на разных сидах и печатает счёт, число кадров и статистику прыжков.

    uv run python evaluate.py best_genome.pkl
    uv run python evaluate.py best_genome.pkl --seeds 1 2 3 --max-frames 20000
"""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def build_args(opts: argparse.Namespace) -> argparse.Namespace:
    # --live без окна бессмысленен: смотреть нечего.
    headed = opts.headed or opts.live
    return argparse.Namespace(
        browsers=1,
        headless=not headed,
        episode_seconds=opts.seconds,
        max_frames=opts.max_frames,
        clear_ms=900,
        poll_interval=0.15,
        route_seed=opts.seeds[0],
        live=opts.live,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка обученного генома NEAT")
    parser.add_argument("model", nargs="?", default="best_genome.pkl", help="файл генома")
    parser.add_argument("--config", default="neat_config.ini", help="конфиг NEAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 777, 424242])
    parser.add_argument("--seconds", type=float, default=600.0, help="лимит эпизода, c")
    parser.add_argument("--max-frames", type=int, default=40000, help="лимит кадров")
    parser.add_argument(
        "--live", action="store_true", help="играть в реальном времени (нужен --headed)"
    )
    parser.add_argument("--headed", action="store_true", help="показать окно Chromium")
    return parser.parse_args()


async def main() -> None:
    opts = parse_args()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        opts.config,
    )
    genome = load_genome(opts.model)
    args = build_args(opts)

    if not args.headless:
        print(f"Открываю окно Chromium: сиды {opts.seeds}, {opts.seconds:g} c на сид.")
    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, args)
        await worker.start()
        try:
            for seed in opts.seeds:
                worker.agent_cfg["seed"] = seed
                result = await worker.run_genome(genome)
                print(
                    f"сид={seed}: статус={result.status} score={result.score} "
                    f"кадров={result.frames} прыжков={result.jumps} "
                    f"лишних={result.useless} приседов={result.ducks}"
                )
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
