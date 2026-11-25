"""Трасса падения с включённым рефлексом приседа."""

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
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(777, 60000))
        await worker.start()
        try:
            r = await worker.run_genome(genome)
            print(f"score={r.score} frames={r.frames} {r.status} "
                  f"jumps={r.jumps} duck={r.ducks}")
            for row in r.trace[-70:]:
                print(
                    f"f={row['f']} ot={row['ot']} oy={row['oy']} ow={row['ow']} "
                    f"dx={row['dx']} dy={row['dy']} gr={row['gr']} dk={row['dk']} "
                    f"jv={row['jv']} dv={row['dv']}"
                )
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())