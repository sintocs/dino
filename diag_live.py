"""Диагностика live-режима: видимое окно + штатная отрисовка игры.

Запускает live-эпизод, во время бега снимает скриншоты окна и счётчик,
затем сравнивает итоговый score с детерминированным simulate().
"""

import argparse
import asyncio
import sys

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome


def make_args(headless: bool, live: bool, max_frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1,
        headless=headless,
        episode_seconds=1000.0,
        max_frames=max_frames,
        clear_ms=900,
        poll_interval=0.05,
        route_seed=777,
        live=live,
        trace=0,
        render=live,
    )


async def main() -> None:
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")

    async with async_playwright() as playwright:
        # 1) Эталон: детерминированная симуляция без отрисовки.
        ref_worker = BrowserWorker(0, playwright, config, make_args(True, False, max_frames))
        await ref_worker.start()
        try:
            ref = await ref_worker.run_genome(genome)
            print(f"reference: status={ref.status} score={ref.score} frames={ref.frames}")
        finally:
            await ref_worker.close()

        # 2) Live: видимое окно, штатная отрисовка.
        worker = BrowserWorker(0, playwright, config, make_args(False, True, max_frames))
        await worker.start()
        try:
            assert worker.page is not None
            live_task = asyncio.create_task(worker.run_genome(genome))

            for i in range(shots):
                await asyncio.sleep(1.0)
                snap = await worker.page.evaluate("() => window.__dino.poll()")
                info = await worker.page.evaluate(
                    """() => {
                        const c = document.querySelector('canvas');
                        const panel = document.getElementById('__dino_demo_overlay');
                        const r = window.Runner.instance_;
                        return {
                            canvasVisibility: c ? getComputedStyle(c).visibility : null,
                            canvasW: c ? c.width : 0,
                            canvasH: c ? c.height : 0,
                            overlay: panel ? panel.innerText.replace(/\\n/g, ' | ') : null,
                            tRexX: r.tRex ? Math.round(r.tRex.xPos) : null,
                            tRexY: r.tRex ? Math.round(r.tRex.yPos) : null,
                            obstacles: r.horizon ? r.horizon.obstacles.length : null,
                            distanceRan: Math.round(r.distanceRan || 0),
                        };
                    }"""
                )
                await worker.page.screenshot(path=f"live_shot_{i}.png")
                print(f"t={i + 1}s state={snap.get('state')} info={info}")

            result = await live_task
            print(f"live: status={result.status} score={result.score} frames={result.frames}")
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
