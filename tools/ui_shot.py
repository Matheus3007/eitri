"""Screenshot pages of `eitri ui` with a headless Chromium inside WSL (never the user's Windows browser).

    ./eitri.sh python tools/ui_shot.py out.png "#project=..&run=..&tab=viewer&anim=1" [--wait 4] [--size 1500x950]

Prints the page's console errors. Needs the dev dependency playwright + `python -m playwright install chromium`.
"""

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

# Chromium's system libraries (nss, nspr, gbm, asound) unpacked without sudo: `apt-get download` + `dpkg -x`
# into ~/opt/chromium-libs/root (see CLAUDE.md)
_LIBS = Path.home() / "opt" / "chromium-libs" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
if _LIBS.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{_LIBS}:{_LIBS / 'nss'}:" + os.environ.get("LD_LIBRARY_PATH", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("hash", nargs="?", default="")
    ap.add_argument("--base", default="http://127.0.0.1:8765/")
    ap.add_argument("--wait", type=float, default=4.0)
    ap.add_argument("--size", default="1500x950")
    ap.add_argument("--click", action="append", default=[], help="CSS selector to click before the shot")
    ap.add_argument("--scroll", default="", help="CSS selector to scroll into view before the shot")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page(viewport={"width": w, "height": h})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(a.base + (a.hash if a.hash.startswith("#") or not a.hash else "#" + a.hash))
        page.wait_for_timeout(a.wait * 1000)
        for sel in a.click:
            page.click(sel)
            page.wait_for_timeout(1500)
        if a.scroll:
            page.locator(a.scroll).scroll_into_view_if_needed()
            page.wait_for_timeout(500)
        page.screenshot(path=a.out)
        b.close()
    print(a.out)
    for e in errors:
        print("console:", e)


if __name__ == "__main__":
    main()
