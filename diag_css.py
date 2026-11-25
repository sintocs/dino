"""Почему ширина canvas не масштабируется: max-width и правила CSS."""

import argparse
import asyncio

import neat
from playwright.async_api import async_playwright

from dino import BrowserWorker, load_genome

JS = """() => {
    const r = window.Runner.instance_;
    const c = r.canvas;
    const cs = getComputedStyle(c);
    const rules = [];
    for (const sheet of Array.from(document.styleSheets)) {
        let list = null;
        try { list = sheet.cssRules; } catch (e) { continue; }
        if (!list) continue;
        for (const rule of Array.from(list)) {
            const t = rule.cssText || '';
            if (t.indexOf('canvas') >= 0 || t.indexOf('max-width') >= 0) {
                rules.push(t.slice(0, 200));
            }
        }
    }
    return {
        maxW: cs.maxWidth, maxH: cs.maxHeight,
        minW: cs.minWidth, boxSizing: cs.boxSizing,
        cssW: cs.width, cssH: cs.height,
        rules: rules.slice(0, 25),
        bodyRules: (() => {
            const out = [];
            for (const sheet of Array.from(document.styleSheets)) {
                let list = null;
                try { list = sheet.cssRules; } catch (e) { continue; }
                if (!list) continue;
                for (const rule of Array.from(list)) {
                    if (rule.selectorText && rule.selectorText.indexOf('canvas') >= 0) {
                        out.push(rule.selectorText + ' => ' + rule.style.cssText.slice(0, 160));
                    }
                }
            }
            return out;
        })(),
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
            await task
        finally:
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
