"""Диагностика: почему видимое окно даёт другой score, чем headless-проверка.

Прогоняет один и тот же геном и сид в четырёх комбинациях:
headless/headed x render=False/True и печатает реальные размеры игрового поля.
"""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


async def run_case(config, genome, headless: bool, render: bool):
    args = argparse.Namespace(
        browsers=1,
        headless=headless,
        episode_seconds=1000.0,
        max_frames=70000,
        clear_ms=900,
        poll_interval=0.05,
        route_seed=777,
        live=False,
        trace=0,
        render=render,
    )
    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, args)
        await worker.start()
        try:
            result = await worker.run_genome(genome)
            info = await worker.page.evaluate(
                """() => {
                    const r = window.Runner.instance_;
                    return {
                        w: r.dimensions.WIDTH,
                        h: r.dimensions.HEIGHT,
                        canvasW: r.canvas.width,
                        canvasH: r.canvas.height,
                        hidden: document.hidden,
                        playing: r.playing,
                        paused: r.paused,
                        crashed: r.crashed
                    };
                }"""
            )
            print(
                f"headless={headless!s:<5} render={render!s:<5} "
                f"status={result.status:<6} score={result.score:<6} "
                f"frames={result.frames:<6} {info}"
            )
        finally:
            await worker.close()


async def main() -> None:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")
    for headless in (True, False):
        for render in (False, True):
            await run_case(config, genome, headless, render)


if __name__ == "__main__":
    asyncio.run(main())
