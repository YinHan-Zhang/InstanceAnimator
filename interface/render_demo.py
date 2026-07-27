# -*- coding: utf-8 -*-
"""Render the InstanceAnimator WebUI (with auto-filled demo data) to PDF and video.

PDF  : full-page capture via headless Chromium.
Video: high-resolution frame-by-frame screenshots stitched with ffmpeg
       (Playwright's built-in recorder caps width at 800px, so we capture
        viewport screenshots at full resolution and encode them ourselves).
"""
import os
import time
import glob
import shutil

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "renders")
PDF_PATH = os.path.join(OUT_DIR, "demo.pdf")
MP4_PATH = os.path.join(OUT_DIR, "demo.mp4")
FRAMES_DIR = "/tmp/ia_frames"

# video capture settings
VW, VH = 1920, 1080          # viewport (== output resolution)
FPS = 25
SCROLL_DOWN_S = 9.0          # seconds to scroll down once (slow & smooth)
HOLD_S = 3.0                 # seconds to hold on the playing result at the end

PREFILL_DONE = (
    "document.querySelector('#resultVideo') && "
    "document.querySelector('#resultVideo').currentSrc && "
    "document.querySelector('#dzVideo video') && "
    "document.querySelector('#dzVideo video').currentSrc"
)


def _capture_once(page, frames_dir):
    """Scroll down once (ending with the result video centered), then
    simulate a mouse click on the result video to play it; hold at the end."""
    idx = 0
    seq = []

    # target scroll position that centers the result video in the viewport
    target = page.evaluate(
        "() => {"
        "  const el = document.querySelector('#resultVideo');"
        "  const r = el.getBoundingClientRect();"
        "  const centerDoc = r.top + window.scrollY + r.height / 2;"
        "  const max = document.body.scrollHeight - window.innerHeight;"
        "  return Math.max(0, Math.min(max, centerDoc - window.innerHeight / 2));"
        "}"
    )

    n = max(1, int(FPS * SCROLL_DOWN_S))
    for i in range(n + 1):
        y = target * i / n
        page.evaluate(f"window.scrollTo(0, {y})")
        time.sleep(0.012)  # let the frame paint
        page.screenshot(path=os.path.join(frames_dir, f"frame_{idx:05d}.png"))
        seq.append(idx)
        idx += 1
        time.sleep(max(0, 1 / FPS - 0.03))

    # ---- simulate mouse moving to the result video, then slow-play (fps=4) ----
    time.sleep(0.3)
    loc = page.eval_on_selector(
        "#resultVideo",
        "el => { const r = el.getBoundingClientRect();"
        "        return [r.x + r.width / 2, r.y + r.height / 2]; }",
    )
    page.mouse.move(loc[0], loc[1], steps=25)   # glide the cursor in
    time.sleep(0.25)
    page.evaluate("window.startSlow && window.startSlow(document.querySelector('#resultVideo'), 4)")

    # ---- hold so the slow-playing result is captured ----
    hold_n = max(1, int(FPS * HOLD_S))
    for _ in range(hold_n):
        page.screenshot(path=os.path.join(frames_dir, f"frame_{idx:05d}.png"))
        seq.append(idx)
        idx += 1
        time.sleep(1 / FPS)
    return seq


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ---------------- PDF (full page, hi-dpi) ----------------
        # Use SCREEN media so the layout matches what's on screen (print media
        # triggers @media print rules that reflow / hide elements -> misalignment).
        page = browser.new_page(viewport={"width": 1440, "height": 1024})
        page.emulate_media(media="screen")
        page.goto(URL, wait_until="networkidle")
        try:
            page.wait_for_function(PREFILL_DONE, timeout=20000)
        except Exception:
            print("warn: prefill wait timed out, continuing...")
        time.sleep(2.5)
        full_h = page.evaluate("document.body.scrollHeight")
        # paper width must exactly match the rendered viewport width (1440px)
        # so content is not offset/clipped horizontally.
        page.pdf(path=PDF_PATH, print_background=True,
                 prefer_css_page_size=False,
                 width="1440px", height=f"{full_h}px",
                 margin={"top": "0", "bottom": "0",
                         "left": "0", "right": "0"})
        print("PDF saved ->", PDF_PATH)
        page.close()

        # ---------------- Video (hi-res frame capture) ----------------
        page = browser.new_page(viewport={"width": VW, "height": VH},
                                device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        try:
            page.wait_for_function(PREFILL_DONE, timeout=20000)
        except Exception:
            pass
        time.sleep(2.5)
        seq = _capture_once(page, FRAMES_DIR)
        page.close()
        browser.close()

    # encode frames -> mp4
    pattern = os.path.join(FRAMES_DIR, "frame_%05d.png")
    rc = os.system(
        f"ffmpeg -y -framerate {FPS} -i '{pattern}' "
        f"-c:v libx264 -pix_fmt yuv420p -crf 17 "
        f"-movflags +faststart '{MP4_PATH}' -loglevel error"
    )
    if rc == 0 and os.path.exists(MP4_PATH):
        print(f"MP4 saved -> {MP4_PATH}  ({len(seq)} frames, {VW}x{VH})")
    else:
        print("ffmpeg encoding failed.")
    # cleanup temp frames
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)


if __name__ == "__main__":
    run()
