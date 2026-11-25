"""Валидация сохранённого генома на нескольких независимых трассах.

Запускает детерминированную симуляцию (без анимации) для одного и того же
генома на разных сидах и печатает счёт, число кадров и статистику прыжков.

    uv run python evaluate.py best_genome.pkl
    uv run python evaluate.py best_genome.pkl --seeds 1 2 3 --max-frames 20000
    uv run python evaluate.py best_genome.pkl --seconds 0 --max-frames 200000 --trace 40
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
        trace=opts.trace,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка обученного генома NEAT")
    parser.add_argument("model", nargs="?", default="best_genome.pkl", help="файл генома")
    parser.add_argument("--config", default="neat_config.ini", help="конфиг NEAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 777, 424242])
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="лимит эпизода, c (0 — без лимита по времени)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="лимит кадров (0 — без лимита; остаётся страховка в run_genome)",
    )
    parser.add_argument(
        "--live", action="store_true", help="играть в реальном времени (нужен --headed)"
    )
    parser.add_argument("--headed", action="store_true", help="показать окно Chromium")
    parser.add_argument(
        "--trace",
        type=int,
        default=0,
        help="показать N последних кадров перед падением",
    )
    return parser.parse_args()


def print_trace(result: "object") -> None:
    """Печатает последние кадры эпизода — видно, какую ситуацию геном не понял."""
    trace = getattr(result, "trace", None) or []
    if not trace:
        return
    print("  последние кадры (f=кадр, dx=до препятствия, dy=высота дино,")
    print("  gr=прыжок, dk=присед, sp=скорость, jv/dv=выходы сети):")
    for row in trace:
        print(
            f"    f={row.get('f'):>6} dx={str(row.get('dx')):>5} "
            f"dy={str(row.get('dy')):>4} gr={row.get('gr')} dk={row.get('dk')} "
            f"sp={row.get('sp')} {str(row.get('ot')):>4} "
            f"oy={str(row.get('oy')):>4} ow={str(row.get('ow')):>4} "
            f"oh={str(row.get('oh')):>4} jv={row.get('jv')} dv={row.get('dv')}"
        )


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
                if result.error:
                    print(f"  ошибка: {result.error}")
                print_trace(result)
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())