"""Дамп самого canvas во время live-бега (без DOM-наложений)."""

import argparse
import asyncio
import base64

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
    return {
        url: c.toDataURL('image/png'),
        tRex: {x: r.tRex.xPos, y: r.tRex.yPos, jumping: r.tRex.jumping,
               status: r.tRex.status, ducking: r.tRex.ducking,
               currentFrame: r.tRex.currentFrame,
               animLen: r.tRex.currentAnimFrames ? r.tRex.currentAnimFrames.length : -1},
        playing: r.playing,
        obstacles: r.horizon.obstacles.map(o => ({x: Math.round(o.xPos), w: o.width, y: Math.round(o.yPos)})),
        horizonY: r.horizon.horizonLine ? r.horizon.horizonLine.yPos : null,
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
            task = asyncio.create_task(worker.run_genome(genome))
            for i in range(4):
                await asyncio.sleep(0.7)
                probe = await worker.page.evaluate(PROBE)
                url = probe.pop("url")
                raw = base64.b64decode(url.split(",", 1)[1])
                with open(f"canvas_{i}.png", "wb") as handle:
                    handle.write(raw)
                print(f"t={i}", probe)
            res = await task
            print("result:", res.status, res.score, res.frames)
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
