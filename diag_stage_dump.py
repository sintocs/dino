"""Проверка canvas: класс, инлайновые стили, применённые правила."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome

JS = """() => {
    const r = window.Runner.instance_;
    const c = r.canvas;
    const cs = getComputedStyle(c);
    return {
        cls: c.className,
        id: c.id,
        inline: c.getAttribute('style'),
        attrW: c.width, attrH: c.height,
        cssW: cs.width, cssH: cs.height,
        cssPos: cs.position, cssTransform: cs.transform,
        rect: (b => ({x: b.x, y: b.y, w: b.width, h: b.height}))(c.getBoundingClientRect()),
        dpr: window.devicePixelRatio,
        dims: r.dimensions,
        styles: Array.from(document.querySelectorAll('style')).map(s => s.id + ':' + s.textContent.slice(0, 120)),
        h1: (h => h ? {display: getComputedStyle(h).display, text: h.innerText} : null)(document.querySelector('h1')),
        bodyCls: document.body.className,
    };
}"""


def make_args(max_frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        browsers=1, headless=False, episode_seconds=1000.0,
        max_frames=max_frames, clear_ms=900, poll_interval=0.05,
        route_seed=777, live=True, trace=0, render=True, live_speed=1.0,
    )


async def main() -> None:
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini",
    )
    genome = load_genome("best_genome_robust.pkl")
    async with async_playwright() as playwright:
        worker = BrowserWorker(0, playwright, config, make_args(600))
        await worker.start()
        try:
            task = asyncio.create_task(worker.run_genome(genome))
            await asyncio.sleep(4)
            info = await worker.page.evaluate(JS)
            for k, v in info.items():
                print(f"{k}: {v}")
            await worker.page.screenshot(path="stage_dump.png")
            await task
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
