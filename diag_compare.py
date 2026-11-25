"""Сравнение трассы с рефлексом приседа и без него (для отладки)."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(seed: int, frames: int, reflex: bool) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=True, episode_seconds=0.0,
        max_frames=frames, clear_ms=900, poll_interval=0.05,
        route_seed=seed, live=False, trace=400, render=False,
        live_speed=1.0, ptero_reflex=reflex,
    )


async def main() -> None:
    frames = 58836
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(777, frames, False))
        await worker.start()
        try:
            result = await worker.run_genome(genome)
            print(f"score={result.score} frames={result.frames} {result.status} "
                  f"duck={result.ducks} jumps={result.jumps}")
            for row in result.trace[-26:]:
                print(
                    f"f={row['f']} ot={row['ot']} oy={row['oy']} dx={row['dx']} "
                    f"dy={row['dy']} gr={row['gr']} dk={row['dk']} "
                    f"jv={row['jv']} dv={row['dv']}"
                )
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())