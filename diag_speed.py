"""Проверка темпа и счёта на обычной скорости (1x)."""

import argparse
import asyncio
import time

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(headless: bool, live: bool, max_frames: int, speed: float) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=headless, episode_seconds=1000.0,
        max_frames=max_frames, clear_ms=900, poll_interval=0.05,
        route_seed=777, live=live, trace=0, render=live, live_speed=speed,
    )


async def main() -> None:
    frames = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 3000
    speed = float(__import__("sys").argv[2]) if len(__import__("sys").argv) > 2 else 1.0
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        ref_worker = BrowserWorker(0, playwright, config, make_args(True, False, frames, speed))
        await ref_worker.start()
        try:
            ref = await ref_worker.run_genome(genome)
            print(f"reference: status={ref.status} score={ref.score} frames={ref.frames}")
        finally:
            await ref_worker.close()

        worker = BrowserWorker(0, playwright, config, make_args(False, True, frames, speed))
        await worker.start()
        try:
            started = time.monotonic()
            result = await worker.run_genome(genome)
            wall = time.monotonic() - started
            print(f"live speed={speed}: status={result.status} score={result.score} "
                  f"frames={result.frames} wall={wall:.1f}s "
                  f"expected={frames / 60.0 / speed:.1f}s")
            await worker.page.screenshot(path="speed_check.png")
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
