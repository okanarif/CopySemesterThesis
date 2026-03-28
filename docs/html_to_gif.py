"""
Convert Plotly HTML animation files to GIF.
Uses Playwright headless Chromium to render and capture frames.
"""
import asyncio
import os
import sys
import glob
from pathlib import Path
from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).parent.parent
SWARM_DIR = REPO_ROOT / "uavsafeplanning" / "SWARM"
OUT_DIR = Path(__file__).parent

TASKS = [
    {
        "html": SWARM_DIR / "simulation.html",
        "gif": OUT_DIR / "simulation.gif",
        "label": "simulation",
    },
    {
        "html": SWARM_DIR / "updated_simulation.html",
        "gif": OUT_DIR / "updated_simulation.gif",
        "label": "updated_simulation",
    },
]

FRAME_RATE = 8        # frames per second for GIF
DURATION_SEC = 12     # total seconds to record
VIEWPORT_W = 1100
VIEWPORT_H = 750


async def capture_frames(html_path: Path, frames_dir: Path, n_frames: int, fps: int):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        await page.goto(f"file://{html_path.resolve()}", wait_until="networkidle", timeout=30000)
        print(f"  Page loaded: {html_path.name}")

        # Wait for Plotly to render
        await page.wait_for_timeout(3000)

        # Set camera to top-down XY-plane view (looking along -Z axis)
        await page.evaluate("""
            () => {
                const divs = document.querySelectorAll('.plotly-graph-div');
                divs.forEach(div => {
                    Plotly.relayout(div, {
                        'scene.camera': {
                            eye:    { x: 0, y: 0, z: 2.5 },
                            up:     { x: 0, y: 1, z: 0   },
                            center: { x: 0, y: 0, z: 0   }
                        }
                    });
                });
            }
        """)
        await page.wait_for_timeout(800)
        print("  Camera set to XY top-down view")

        # Try to click the Play button if present
        try:
            # Plotly animation play button selector
            play_btn = page.locator('[data-attr="updatemenu-button"]').first
            if await play_btn.is_visible():
                await play_btn.click()
                print("  Clicked Play button")
            else:
                # Try generic play text button
                btn = page.get_by_text("▶", exact=False).first
                if await btn.is_visible():
                    await btn.click()
                    print("  Clicked ▶ button")
        except Exception as e:
            print(f"  No play button found ({e}), capturing static/manual frames")

        interval_ms = int(1000 / fps)
        os.makedirs(frames_dir, exist_ok=True)

        for i in range(n_frames):
            frame_path = frames_dir / f"frame_{i:04d}.png"
            await page.screenshot(path=str(frame_path), clip={
                "x": 0, "y": 0, "width": VIEWPORT_W, "height": VIEWPORT_H
            })
            if i % fps == 0:
                print(f"  Frame {i+1}/{n_frames}")
            await page.wait_for_timeout(interval_ms)

        await browser.close()
    print(f"  {n_frames} frames captured.")


def frames_to_gif(frames_dir: Path, output_gif: Path, fps: int):
    pattern = str(frames_dir / "frame_%04d.png")
    palette = str(frames_dir / "palette.png")

    # Generate palette for better quality
    ret = os.system(
        f'ffmpeg -y -framerate {fps} -i "{pattern}" '
        f'-vf "fps={fps},scale=960:-1:flags=lanczos,palettegen" "{palette}" 2>/dev/null'
    )
    if ret != 0:
        print("  Palette generation failed, trying direct conversion.")
        os.system(
            f'ffmpeg -y -framerate {fps} -i "{pattern}" '
            f'-vf "fps={fps},scale=960:-1:flags=lanczos" -loop 0 "{output_gif}" 2>/dev/null'
        )
        return

    # Use palette to create GIF
    os.system(
        f'ffmpeg -y -framerate {fps} -i "{pattern}" -i "{palette}" '
        f'-lavfi "fps={fps},scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse" '
        f'-loop 0 "{output_gif}" 2>/dev/null'
    )


async def main():
    for task in TASKS:
        html_path = task["html"]
        gif_path = task["gif"]
        label = task["label"]

        if not html_path.exists():
            print(f"[SKIP] {html_path} not found.")
            continue

        frames_dir = OUT_DIR / f"_frames_{label}"
        n_frames = DURATION_SEC * FRAME_RATE

        print(f"\n=== Processing: {label} ===")
        print(f"  HTML: {html_path}")
        print(f"  GIF:  {gif_path}")
        print(f"  Capturing {n_frames} frames at {FRAME_RATE} fps...")

        await capture_frames(html_path, frames_dir, n_frames, FRAME_RATE)

        print("  Converting frames to GIF...")
        frames_to_gif(frames_dir, gif_path, FRAME_RATE)

        if gif_path.exists():
            size_kb = gif_path.stat().st_size // 1024
            print(f"  GIF created: {gif_path} ({size_kb} KB)")
        else:
            print(f"  ERROR: GIF not created at {gif_path}")

        # Clean up temp frames
        for f in frames_dir.glob("*.png"):
            f.unlink()
        frames_dir.rmdir()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
