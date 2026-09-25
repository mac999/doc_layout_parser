import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageStat
from playwright.sync_api import expect, sync_playwright


def record(url, destination, check_only=False):
    frames = []
    durations = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def capture(duration=1400):
            page.wait_for_function("[...document.images].every(image => image.complete)")
            assert page.evaluate("[...document.images].every(image => image.naturalWidth > 0)")
            frame = Image.open(BytesIO(page.screenshot())).convert("RGB")
            assert sum(ImageStat.Stat(frame).var) > 100, "Blank screenshot"
            frames.append(frame)
            durations.append(duration)

        def selected(region_id):
            expect(page.locator(f'.rrow[data-id="{region_id}"]')).to_have_attribute("aria-selected", "true")
            expect(page.locator(f'.rgn[data-id="{region_id}"]')).to_have_class("rgn sel")
            page.wait_for_function("""() => {
                const row = document.querySelector('.rrow.sel').getBoundingClientRect();
                const list = document.querySelector('#rlist').getBoundingClientRect();
                return row.top >= list.top - 1 && row.bottom <= list.bottom + 1;
            }""")

        def snapshot():
            return page.evaluate("({file:state.file,page:state.page,selected:state.selId,view:state.view,"
                                 "types:state.types,base:state.base,layers:state.layers,text:state.layout.regions})")

        page.goto(url)
        expect(page.locator("#rlist .rrow").first).to_be_visible()
        expect(page.locator("#regionHeading")).to_have_text("레이아웃 영역")
        capture()
        page.locator('[data-lang="en"]').click()
        expect(page.locator("#regionHeading")).to_have_text("Layout Regions")
        expect(page.locator('[data-base="page"]')).to_have_text("Original")
        capture()

        target = page.locator("#rgnLayer .rgn").last
        region_id = target.get_attribute("data-id")
        target.click()
        selected(region_id)
        expect(page.locator("#selection")).to_have_text(f"Selected {region_id}")
        capture(2200)
        page.locator(".zoombtn").click()
        capture(1600)
        before = snapshot()
        page.locator('[data-lang="ko"]').click()
        assert snapshot() == before, "Language switch changed document state"
        selected(region_id)
        expect(page.locator("#selection")).to_have_text(f"선택 {region_id}")
        capture(1800)

        bounds = page.locator("#canvasWrap").bounding_box()
        assert bounds is not None
        center_x = bounds["x"] + bounds["width"] / 2
        center_y = bounds["y"] + bounds["height"] / 2
        page.mouse.move(center_x, center_y)
        page.mouse.down()
        page.mouse.move(center_x + 40, center_y + 25, steps=5)
        page.mouse.up()
        assert page.evaluate("state.selId") == region_id, "Pan changed selection"
        assert snapshot()["view"] != before["view"], "Pan did not move the canvas"
        page.locator("#zoomFit").click()
        previous_row = page.locator("#rlist .rrow").nth(-2)
        previous_row.focus()
        previous_row.press("Enter")
        selected(previous_row.get_attribute("data-id"))

        page.locator('[data-lang="en"]').click()
        page.reload()
        expect(page.locator("#regionHeading")).to_have_text("Layout Regions")
        expect(page.locator("#rlist .rrow").first).to_be_visible()
        file_item = page.locator('.fitem[data-file="img3"]')
        assert file_item.count(), "Demo requires the img3 sample output"
        file_item.locator(".fhead").click()
        file_item.locator(".pitem").first.click()
        page.wait_for_function("state.file === 'img3' && state.layout?.has_native_vectors")
        capture()
        page.locator("#lyVec").click()
        expect(page.locator("#vecLayer path").first).to_be_attached()
        vector_id = page.evaluate("state.layout.regions.find(region => region.vector_file).id")
        page.locator(f'.rrow[data-id="{vector_id}"]').click()
        selected(vector_id)
        page.locator('button[onclick="setDetailTab(\'vec\')"]').click()
        expect(page.locator("#mediaBox svg")).to_be_visible()
        capture(2400)
        page.locator(".zoombtn").click()
        capture(1800)
        bounds = page.locator("#canvasWrap").bounding_box()
        assert bounds is not None
        page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
        for _ in range(5):
            page.mouse.wheel(0, 100)
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            capture(120)
        page.locator("#zoomFit").click()
        page.locator("#lyNative").click()
        expect(page.locator("#nativeLayer path")).to_be_attached()
        capture(1800)
        page.locator('.pitem[data-file="img3"]').nth(1).click()
        page.wait_for_function("state.page === 'page_002' && state.layout?.page === 2")
        capture(1800)
        page.locator('[data-base="overlay"]').click()
        capture(1800)
        page.locator('[data-lang="ko"]').click()
        capture(1800)

        for width, height in [(390, 844), (800, 900), (1600, 1000)]:
            page.set_viewport_size({"width": width, "height": height})
            page.locator("#zoomFit").click()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"
            assert page.evaluate("""() => {
                const toolbar = document.querySelector('#toolbar').getBoundingClientRect();
                const canvas = document.querySelector('#canvasWrap').getBoundingClientRect();
                return toolbar.bottom <= canvas.top + 1 && canvas.height > 100;
            }"""), "Toolbar overlaps canvas"
            expect(page.locator('[data-lang="en"]')).to_be_visible()
            capture(400)
            frames.pop()
            durations.pop()
        assert not errors, errors
        browser.close()

    if not check_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
        palette_frames = [frame.quantize(colors=192) for frame in frames]
        palette_frames[0].save(destination, save_all=True, append_images=palette_frames[1:],
                               duration=durations, loop=0, optimize=True, disposal=2)
        with Image.open(destination) as animation:
            assert animation.n_frames > 1 and animation.size == (1280, 800)
        print(f"Demo saved: {destination} ({destination.stat().st_size:,} bytes)")
    print("Viewer checks passed: language, persistence, selection, pan, vectors, assets, responsive layout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check viewer interactions and record the sample demo.")
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "doc" / "viewer-demo.gif")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    record(args.url, args.output, args.check_only)