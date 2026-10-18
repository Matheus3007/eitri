"""`eitri ui`: a local web app over the same files the CLI writes (nothing here is a second source of truth).

    GET  /api/projects                 projects with their spec summary
    GET  /api/project?name=            spec text + runs
    GET  /api/run?path=                a run: report headline, checks, comparison, images, animations
    POST /api/anim?path=&source=       build (once, cached in the run) an animation: fea[:<mat>] | explicit:<dir> |
                                       topo:<dir>; returns its folder
    PUT  /api/spec?name=               save a spec (validated first)
    POST /api/job                      {"cmd": eval|optimize|topo|impact|compare|refine, "project", "args": [...]}
    GET  /api/jobs, /api/job?id=       background jobs (the eitri CLI in a subprocess) and their logs
    GET  /files/<path>                 any file under projects/
Standard library only; the page loads three.js from a CDN.
"""

import json
import subprocess
import threading
import time
import tomllib
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / "projects"
STATIC = Path(__file__).with_name("static")
JOBS: dict[int, dict] = {}
_job_lock = threading.Lock()


def _safe(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(PROJECTS.resolve())):
        raise PermissionError(rel)
    return p


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


def projects() -> list[dict]:
    out = []
    for d in sorted(PROJECTS.iterdir()):
        spec = d / "spec.toml"
        if not spec.exists():
            continue
        try:
            t = tomllib.loads(spec.read_text())
        except Exception as e:  # show broken specs too
            t = {"name": d.name, "_error": str(e)}
        runs = sorted((d / "runs").glob("*")) if (d / "runs").exists() else []
        out.append({"name": d.name, "title": t.get("name", d.name), "type": t.get("weapon", {}).get("type"),
                    "reference": (t.get("reference") or {}).get("spec"), "n_runs": len(runs),
                    "last_run": runs[-1].name if runs else None, "error": t.get("_error")})
    return out


def _headline(report: dict) -> dict:
    r = (report.get("results") or [report])[0]
    if "error" in r:
        return {"ok": False, "error": r["error"]}
    m, h, s = r.get("mass", {}), r.get("impact", {}), r.get("stress", {})
    return {"ok": r.get("ok"), "failed": r.get("failed", []), "material": r.get("material"),
            "mass_g": m.get("mass_kg", 0) * 1e3, "energy_j": h.get("energy_j"), "moi_gm2": m.get("moi_kgm2", 0) * 1e3,
            "imbalance_gmm": m.get("static_imbalance_gmm"), "sf_spin": s.get("safety_factor_yield"),
            "sf_impact": s.get("impact_safety_factor"), "yield_force_kn": s.get("yield_force_kn")}


def project(name: str) -> dict:
    d = PROJECTS / name
    runs = []
    for r in sorted((d / "runs").glob("*"), reverse=True) if (d / "runs").exists() else []:
        if not r.is_dir():
            continue
        info = {"path": _rel(r), "name": r.name, "kind": r.name.split("-")[-1]}
        for rep in (r / "report.json", r / "best" / "report.json"):
            if rep.exists():
                try:
                    info |= _headline(json.loads(rep.read_text()))
                except Exception:
                    pass
                break
        runs.append(info)
    return {"name": name, "spec": (d / "spec.toml").read_text(), "runs": runs}


def _anim_sources(run: Path) -> list[dict]:
    out = []
    for fea in sorted(run.glob("**/fea/*/impact.frd")):
        out.append({"source": f"fea:{_rel(fea.parent)}", "label": f"static FEA ({fea.parent.name}) "
                    f"{_rel(fea.parent).split('/runs/')[-1]}"})
    for v in sorted(run.glob("**/vtk/*.pvd")):
        out.append({"source": f"explicit:{_rel(v.parent)}", "label": f"explicit hit {_rel(v.parent.parent).split('/runs/')[-1]}"})
    for t in sorted(run.glob("**/density_history.npz")):
        out.append({"source": f"topo:{_rel(t)}", "label": f"topology {t.parent.name}"})
    for a in sorted(run.glob("**/anim/*/manifest.json")):
        out.append({"source": "ready", "anim": _rel(a.parent), "label": json.loads(a.read_text()).get("title")})
    return out


def run_info(rel: str) -> dict:
    r = _safe(rel)
    info = {"path": rel, "name": r.name, "files": [], "images": [], "markdown": {}, "anims": _anim_sources(r)}
    for f in sorted(r.rglob("*")):
        if f.is_file() and "fea" not in f.parts and "vtk" not in f.parts and "anim" not in f.parts:
            info["files"].append(_rel(f))
            if f.suffix == ".png":
                info["images"].append(_rel(f))
            if f.suffix == ".md":
                info["markdown"][_rel(f)] = f.read_text()
    for rep in (r / "report.json", r / "best" / "report.json"):
        if rep.exists():
            data = json.loads(rep.read_text())
            info["headline"] = _headline(data)
            res = (data.get("results") or [data])[0]
            info["checks"] = res.get("checks", [])
            info["design"] = res.get("design")
            break
    for summ in ("summary.json", "topo.json", "impact.json"):
        if (r / summ).exists():
            info[summ.split(".")[0]] = json.loads((r / summ).read_text())
    return info


def build_anim(rel: str, source: str) -> str:
    from eitri.ui import anim

    r = _safe(rel)
    kind, _, arg = source.partition(":")
    if kind == "fea":
        fea_dir = _safe(arg)
        out = fea_dir.parent.parent / "anim" / f"fea_{fea_dir.name}"
        if not (out / "manifest.json").exists():
            rep_dir = fea_dir.parent.parent
            rep = json.loads((rep_dir / "report.json").read_text())
            res = next(x for x in rep["results"] if x.get("material") == fea_dir.name)
            from eitri import materials

            s = res["stress"]
            anim.from_fea(fea_dir, out, s["yield_force_kn"], s["impact_at_estimated_peak"]["force_kn"],
                          materials.get(fea_dir.name).yield_)
    elif kind == "explicit":
        vdir = _safe(arg)
        out = vdir.parent / "anim" / "explicit"
        if not (out / "manifest.json").exists():
            anim.from_explicit(vdir, out, dt_ms=_explicit_dt(vdir), title=f"explicit hit: {vdir.parent.name}")
    elif kind == "topo":
        npz = _safe(arg)
        out = npz.parent / "anim" / "topo"
        if not (out / "manifest.json").exists():
            anim.from_topo(npz, out)
    else:
        raise ValueError(source)
    return _rel(out)


def _explicit_dt(vdir: Path) -> float:
    import re

    pvd = next(vdir.glob("*.pvd"), None)
    if pvd:
        ts = [float(x) for x in re.findall(r'timestep="([^"]+)"', pvd.read_text())]
        if len(ts) > 1:
            return ts[1] - ts[0]
    return 0.01


def save_spec(name: str, text: str) -> dict:
    from eitri.spec import Spec

    try:
        Spec.model_validate(tomllib.loads(text))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    (PROJECTS / name / "spec.toml").write_text(text)
    return {"ok": True}


COMMANDS = {"eval", "optimize", "topo", "impact", "compare", "refine", "check"}


def start_job(cmd: str, proj: str, args: list[str]) -> dict:
    if cmd not in COMMANDS:
        raise ValueError(cmd)
    spec = PROJECTS / proj / "spec.toml"
    log = PROJECTS / proj / "runs" / f"job-{int(time.time())}-{cmd}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ROOT / "eitri.sh"), "eitri", cmd, str(spec.relative_to(ROOT)), *[str(a) for a in args]]
    fh = open(log, "w")
    p = subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    with _job_lock:
        jid = max(JOBS, default=0) + 1
        JOBS[jid] = {"id": jid, "cmd": cmd, "project": proj, "args": args, "log": _rel(log), "proc": p,
                     "started": time.time()}
    return {"id": jid}


def jobs() -> list[dict]:
    out = []
    for j in JOBS.values():
        rc = j["proc"].poll()
        out.append({k: v for k, v in j.items() if k != "proc"} | {"running": rc is None, "returncode": rc})
    return sorted(out, key=lambda j: -j["id"])


def job_log(jid: int, tail: int = 20000) -> dict:
    j = JOBS[jid]
    text = (ROOT / j["log"]).read_text(errors="ignore")
    noise = ("zoxide", "Please ensure", "If the issue", "github.com/ajeet", "Disable this message")
    text = "\n".join(l for l in text.splitlines() if not l.startswith(noise))
    return {"id": jid, "running": j["proc"].poll() is None, "log": text[-tail:]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # one line per request on stderr when EITRI_UI_LOG is set
        import os
        import sys

        if os.environ.get("EITRI_UI_LOG"):
            print(self.address_string(), fmt % args, file=sys.stderr, flush=True)

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj, default=str).encode(), "application/json", code)

    def _query(self):
        u = urllib.parse.urlparse(self.path)
        return u.path, {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        path, q = self._query()
        try:
            if path in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if not str(f).startswith(str(STATIC.resolve())) or not f.exists():
                    return self._json({"error": "not found"}, 404)
                ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8"}.get(
                    f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            if path == "/api/projects":
                return self._json(projects())
            if path == "/api/project":
                return self._json(project(q["name"]))
            if path == "/api/run":
                return self._json(run_info(q["path"]))
            if path == "/api/data":  # animation arrays: Chrome swallows fetches of URLs ending in .bin (a 204
                f = _safe(q["path"])  # that never reaches the server), so they don't go through /files/
                return self._send(f.read_bytes(), "application/x-eitri-array")
            if path == "/api/jobs":
                return self._json(jobs())
            if path == "/api/job":
                return self._json(job_log(int(q["id"])))
            if path.startswith("/files/"):
                f = _safe(urllib.parse.unquote(path[len("/files/"):]))
                if not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = {".png": "image/png", ".json": "application/json", ".md": "text/markdown",
                         ".toml": "text/plain", ".step": "application/step",
                         # not octet-stream: Chrome treats that as a download and fetch() can get an empty 204
                         ".bin": "application/x-eitri-array"}.get(f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            return self._json({"error": "not found"}, 404)
        except PermissionError:
            return self._json({"error": "outside projects/"}, 403)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        path, q = self._query()
        try:
            if path == "/api/anim":
                return self._json({"anim": build_anim(q["path"], q["source"])})
            if path == "/api/job":
                b = json.loads(self._body() or b"{}")
                return self._json(start_job(b["cmd"], b["project"], b.get("args", [])))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_PUT(self):
        path, q = self._query()
        if path == "/api/spec":
            return self._json(save_spec(q["name"], self._body().decode()))
        return self._json({"error": "not found"}, 404)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"eitri ui on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
