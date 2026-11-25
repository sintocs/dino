"""Проверка обычной скорости на длинном прогоне + скриншоты сцены."""

import argparse
import asyncio
import time

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(headless: bool, live: bool, max_frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=headless, episode_seconds=1000.0,
        max_frames=max_frames, clear_ms=900, poll_interval=0.05,
        route_seed=777, live=live, trace=0, render=live, live_speed=1.0,
    )


async def main() -> None:
    frames = 12000
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(False, True, frames))
        await worker.start()
        try:
            task = asyncio.create_task(worker.run_genome(genome))
            started = time.monotonic()
            for i in range(5):
                await asyncio.sleep(30)
                info = await worker.page.evaluate(
                    """() => {
                        const r = window.Runner.instance_;
                        const c = r.canvas.getBoundingClientRect();
                        return {
                            score: Math.round((r.distanceRan || 0) * 0.025),
                            speed: Math.round(r.currentSpeed * 100) / 100,
                            canvasBox: {x: Math.round(c.x), y: Math.round(c.y),
                                        w: Math.round(c.width), h: Math.round(c.height)},
                            overlay: (document.getElementById('__dino_demo_overlay') || {}).innerText,
                        };
                    }"""
                )
                elapsed = time.monotonic() - started
                print(f"t={elapsed:5.1f}s {info}")
                await worker.page.screenshot(path=f"demo_{i}.png")
            res = await task
            print("final:", res.status, res.score, res.frames)
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
