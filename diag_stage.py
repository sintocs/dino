"""Быстрая проверка сцены: размеры canvas, темп, скриншот."""

import argparse
import asyncio
import time

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome

INFO_JS = """() => {
    const r = window.Runner.instance_;
    const c = r.canvas.getBoundingClientRect();
    const host = r.containerEl ? r.containerEl.getBoundingClientRect() : null;
    const h1 = document.querySelector('h1');
    return {
        score: Math.round((r.distanceRan || 0) * 0.025),
        speed: Math.round(r.currentSpeed * 100) / 100,
        frames: Math.round(r.distanceRan / (r.currentSpeed || 1)),
        canvasBox: {x: Math.round(c.x), y: Math.round(c.y),
                    w: Math.round(c.width), h: Math.round(c.height)},
        hostBox: host ? {x: Math.round(host.x), y: Math.round(host.y),
                         w: Math.round(host.width), h: Math.round(host.height)} : null,
        h1Visible: h1 ? getComputedStyle(h1).display !== 'none' : false,
        overlay: (document.getElementById('__dino_demo_overlay') || {}).innerText || '',
        viewport: {w: window.innerWidth, h: window.innerHeight},
    };
}"""


def make_args(max_frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=False, episode_seconds=1000.0,
        max_frames=max_frames, clear_ms=900, poll_interval=0.05,
        route_seed=777, live=True, trace=0, render=True, live_speed=1.0,
    )


async def main() -> None:
    frames = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 1200
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(frames))
        await worker.start()
        try:
            task = asyncio.create_task(worker.run_genome(genome))
            started = time.monotonic()
            for i in range(3):
                await asyncio.sleep(5)
                info = await worker.page.evaluate(INFO_JS)
                print(f"t={time.monotonic() - started:5.1f}s {info}")
                await worker.page.screenshot(path=f"stage_{i}.png")
            res = await task
            wall = time.monotonic() - started
            print(f"final: status={res.status} score={res.score} frames={res.frames} "
                  f"wall={wall:.1f}s expected={frames / 60.0:.1f}s")
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
