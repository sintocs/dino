"""Что реально нарисовано на canvas во время live-прогона."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=False, episode_seconds=1000.0, max_frames=70000,
        clear_ms=900, poll_interval=0.05, route_seed=777, live=True,
        trace=0, render=True,
    )


PROBE = """() => {
    const r = window.Runner.instance_;
    const c = r.canvas;
    const ctx = c.getContext('2d');
    const box = c.getBoundingClientRect();
    let ink = 0, total = 0;
    try {
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        for (let i = 0; i < d.length; i += 4) {
            total++;
            if (d[i] < 200 || d[i+1] < 200 || d[i+2] < 200) { ink++; }
        }
    } catch (e) { ink = -1; }
    return {
        box: {x: box.x, y: box.y, w: box.width, h: box.height},
        canvas: {w: c.width, h: c.height},
        ink, total,
        playing: r.playing, activated: r.activated, crashed: r.crashed,
        distanceRan: Math.round(r.distanceRan),
        tRex: {x: r.tRex.xPos, y: r.tRex.yPos, jumping: r.tRex.jumping, status: r.tRex.status},
        obstacles: r.horizon.obstacles.map(o => ({x: Math.round(o.xPos), y: Math.round(o.yPos), w: o.width})),
        sprite: {
            complete: !!(window.Runner.imageSprite && window.Runner.imageSprite.complete),
            w: window.Runner.imageSprite ? window.Runner.imageSprite.width : 0,
        },
        imgs: Array.from(document.images).map(i => ({
            src: i.currentSrc.split('/').pop(), w: i.width, h: i.height,
            vis: getComputedStyle(i).visibility,
            box: (b => ({x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height)}))(i.getBoundingClientRect())
        })),
    };
}"""


async def main() -> None:
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args())
        await worker.start()
        try:
            assert worker.page is not None
            # Базовая страница до запуска агента.
            await worker.page.screenshot(path="probe_idle.png")
            print("idle:", await worker.page.evaluate(PROBE))

            task = asyncio.create_task(worker.run_genome(genome))
            for i in range(3):
                await asyncio.sleep(1.0)
                print(f"live t={i+1}s:", await worker.page.evaluate(PROBE))
                await worker.page.screenshot(path=f"probe_live_{i}.png")
            res = await task
            print("result:", res.status, res.score, res.frames)
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
