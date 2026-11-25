"""Статистика по прогонам: с рефлексом приседа и без него."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(seed: int, frames: int, reflex: bool) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=True, episode_seconds=0.0,
        max_frames=frames, clear_ms=900, poll_interval=0.05,
        route_seed=seed, live=False, trace=0, render=False,
        live_speed=1.0, ptero_reflex=reflex,
    )


async def main() -> None:
    seeds = [777, 2026, 9001, 12345, 424242]
    frames = 60000
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        for reflex in (True, False):
            print(f"--- ptero_reflex={reflex} ---", flush=True)
            for seed in seeds:
                worker = BrowserWorker(0, playwright, config,
                                       make_args(seed, frames, reflex))
                await worker.start()
                try:
                    r = await worker.run_genome(genome)
                    print(
                        f"seed {seed}: score={r.score} frames={r.frames} "
                        f"status={r.status} jumps={r.jumps} "
                        f"useless={r.useless} duck_frames={r.ducks}",
                        flush=True,
                    )
                finally:
                    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())