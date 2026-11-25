"""Считает по трассе, под птицами какой высоты агент приседал."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(seed: int, frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=True, episode_seconds=0.0,
        max_frames=frames, clear_ms=900, poll_interval=0.05,
        route_seed=seed, live=False, trace=400, render=False,
        live_speed=1.0, ptero_reflex=True,
    )


async def main() -> None:
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_reflex.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(2026, 60000))
        await worker.start()
        try:
            result = await worker.run_genome(genome)
            seen: dict[int, list[int]] = {}
            for row in result.trace:
                if row["ot"] != "PTER":
                    continue
                slot = seen.setdefault(row["oy"], [0, 0])
                slot[0] += 1
                slot[1] += row["dk"]
            print(f"score={result.score} frames={result.frames} {result.status}")
            for y in sorted(seen):
                total, duck = seen[y]
                print(f"ptero yPos={y}: кадров={total} из них присед={duck}")
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())