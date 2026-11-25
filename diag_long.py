"""Длинный показ в реальном темпе: скриншоты каждую минуту + сверка счёта."""

import argparse
import asyncio
import time

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome

INFO_JS = """() => {
    const r = window.Runner.instance_;
    const c = r.canvas.getBoundingClientRect();
    return {
        score: Math.round((r.distanceRan || 0) * 0.025),
        speed: Math.round(r.currentSpeed * 100) / 100,
        frames: window.__dinoFrames ? window.__dinoFrames() : null,
        canvasBox: {x: Math.round(c.x), y: Math.round(c.y),
                    w: Math.round(c.width), h: Math.round(c.height)},
        overlay: (document.getElementById('__dino_demo_overlay') || {}).innerText || '',
    };
}"""


def make_args(headless: bool, live: bool, max_frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=headless, episode_seconds=1000.0,
        max_frames=max_frames, clear_ms=900, poll_interval=0.05,
        route_seed=777, live=live, trace=0, render=live, live_speed=1.0,
    )


async def main() -> None:
    frames = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 30000
    shots = int(__import__("sys").argv[2]) if len(__import__("sys").argv) > 2 else 6
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        ref_worker = BrowserWorker(0, playwright, config, make_args(True, False, frames))
        await ref_worker.start()
        try:
            ref = await ref_worker.run_genome(genome)
            print(f"reference: status={ref.status} score={ref.score} frames={ref.frames}")
        finally:
            await ref_worker.close()

        worker = BrowserWorker(0, playwright, config, make_args(False, True, frames))
        await worker.start()
        try:
            started = time.monotonic()
            task = asyncio.create_task(worker.run_genome(genome))
            interval = frames / 60.0 / shots
            for i in range(shots):
                await asyncio.sleep(interval)
                if task.done():
                    break
                info = await worker.page.evaluate(INFO_JS)
                print(f"t={time.monotonic() - started:6.1f}s {info}", flush=True)
                await worker.page.screenshot(path=f"long_{i}.png")
            res = await task
            wall = time.monotonic() - started
            print(f"live: status={res.status} score={res.score} frames={res.frames} "
                  f"wall={wall:.1f}s expected={frames / 60.0:.1f}s")
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
