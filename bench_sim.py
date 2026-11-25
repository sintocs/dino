import asyncio
import time

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


async def main() -> None:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.ini",
    )
    genome = load_genome("best_genome.pkl")

    import argparse
    args = argparse.Namespace(
        browsers=1,
        headless=True,
        episode_seconds=1200.0,
        max_frames=60000,
        clear_ms=900,
        poll_interval=0.15,
        route_seed=12345,
        live=False,
    )

    async with async_playwright() as pw:
        w = BrowserWorker(0, pw, config, args)
        await w.start()
        t0 = time.monotonic()
        r = await w.run_genome(genome)
        dt = time.monotonic() - t0
        await w.close()
        print(
            f"status={r.status} score={r.score} frames={r.frames} "
            f"wall={dt:.2f}s frames/sec={r.frames / dt:,.0f}"
        )


asyncio.run(main())