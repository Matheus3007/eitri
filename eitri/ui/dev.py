"""Hot reload for `eitri ui`: run the server in a child process and restart it whenever a Python file under
eitri/ changes. The page holds an event stream (/api/events); a new server's boot id makes it reload itself,
and the server itself pushes static-file changes (CSS swapped in place, JS/HTML reload the page)."""

import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def _sources() -> dict[Path, float]:
    out = {}
    for f in PKG.rglob("*.py"):
        try:
            out[f] = f.stat().st_mtime
        except OSError:
            pass
    return out


def run(host: str, port: int) -> None:
    argv = [sys.executable, "-m", "eitri.ui.server", "--host", host, "--port", str(port)]
    child = subprocess.Popen(argv)
    seen = _sources()
    try:
        while True:
            time.sleep(1.0)
            now = _sources()
            changed = [f for f in now if now[f] != seen.get(f)]
            if changed:
                seen = now
                time.sleep(0.3)  # let an editor finish writing
                print(f"[eitri ui] {changed[0].relative_to(PKG.parent)} changed: restarting the server", flush=True)
                child.terminate()
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    child.kill()
                child = subprocess.Popen(argv)
            elif child.poll() is not None:  # crashed (a syntax error mid-edit): wait for the next save
                seen = _sources()
                while _sources() == seen:
                    time.sleep(1.0)
                seen = _sources()
                child = subprocess.Popen(argv)
    except KeyboardInterrupt:
        child.terminate()
