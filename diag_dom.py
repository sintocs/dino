"""Разведка DOM: где живёт надпись Press Space и как устроен контейнер игры."""

import asyncio

from playwright.async_api import async_playwright

from dino import GAME_URL


async def main() -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1000, "height": 500})
        await page.goto(GAME_URL, wait_until="load")
        await page.wait_for_function(
            "() => !!(window.Runner && window.Runner.instance_)", timeout=30000
        )
        info = await page.evaluate(
            """() => {
                const found = [];
                document.querySelectorAll('body *').forEach(el => {
                    const own = Array.from(el.childNodes)
                        .filter(n => n.nodeType === 3)
                        .map(n => n.textContent.trim()).join(' ').trim();
                    if (own && own.length < 60) {
                        found.push({tag: el.tagName, cls: el.className,
                                    text: own,
                                    box: (b => ({x: Math.round(b.x), y: Math.round(b.y),
                                                 w: Math.round(b.width), h: Math.round(b.height)}))(el.getBoundingClientRect())});
                    }
                });
                const r = window.Runner.instance_;
                return {
                    found,
                    outer: r.outerContainerEl ? {tag: r.outerContainerEl.tagName,
                        cls: r.outerContainerEl.className} : null,
                    containerCls: r.containerEl.className,
                    canvasCls: r.canvas.className,
                    bodyCls: document.body.className,
                    arcade: typeof r.setArcadeMode,
                };
            }"""
        )
        for item in info["found"]:
            print(item)
        print({k: v for k, v in info.items() if k != "found"})
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
