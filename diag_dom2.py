"""Структура DOM страницы игры."""

import asyncio

from playwright.async_api import async_playwright

from dino import GAME_URL

DUMP = """() => {
    function walk(el, depth) {
        if (depth > 4) return null;
        return {
            tag: el.tagName,
            cls: typeof el.className === 'string' ? el.className : '',
            id: el.id,
            box: (b => ({x: Math.round(b.x), y: Math.round(b.y),
                         w: Math.round(b.width), h: Math.round(b.height)}))(el.getBoundingClientRect()),
            kids: Array.from(el.children).map(c => walk(c, depth + 1)).filter(Boolean),
        };
    }
    return walk(document.body, 0);
}"""


async def main() -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1000, "height": 500})
        await page.goto(GAME_URL, wait_until="load")
        await page.wait_for_function(
            "() => !!(window.Runner && window.Runner.instance_)", timeout=30000
        )
        import json
        print(json.dumps(await page.evaluate(DUMP), indent=1, ensure_ascii=False))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
