"""`eitri ui`: a local web app over the same files the CLI writes (nothing here is a second source of truth).

    GET  /api/projects                 projects with their spec summary
    GET  /api/project?name=            spec text + runs
    GET  /api/run?path=                a run: report headline, checks, comparison, images, animations
    POST /api/anim?path=&source=       build (once, cached in the run) an animation: fea[:<mat>] | explicit:<dir> |
                                       topo:<dir>; returns its folder
    PUT  /api/spec?name=               save a spec (validated first)
    POST /api/job                      {"cmd": eval|optimize|topo|impact|compare|refine, "project", "args": [...]}
    GET  /api/jobs, /api/job?id=       background jobs (the eitri CLI in a subprocess) and their logs
    GET  /api/catalog                  every run of every project (the Compare selectors)
    GET  /api/timeline?dir=            an animation's manifest + solver steps per frame + time curves
    POST /api/part?step=               tessellate a STEP (once, cached next to it) for the part viewer
    GET  /api/events                   server-sent events: hello (boot id), static (page files), data (projects)
    GET  /files/<path>                 any file under projects/
Standard library only; the page loads three.js from a CDN. `eitri ui` runs it under ui/dev.py, which restarts
it when eitri/ changes; the page reloads itself on a new boot id or a changed static file.
"""

import json
import os
import queue
import subprocess
import threading
import time
import uuid
import tomllib
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / "projects"
STATIC = Path(__file__).with_name("static")
JOBS_FILE = PROJECTS / ".eitri_jobs.json"
_job_lock = threading.Lock()
BOOT_ID = uuid.uuid4().hex[:8]
SKIP_DIRS = {"fea", "vtk", "anim", "__pycache__"}


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
            "sf_impact": s.get("impact_safety_factor"), "yield_force_kn": s.get("yield_force_kn"),
            "rpm": (r.get("drive") or {}).get("top_rpm"), "spinup_s": (r.get("drive") or {}).get("spinup_s_95"),
            "tip_speed_mps": h.get("tip_speed_mps"), "bite_mm": h.get("bite_mm"),
            "effective_mass_kg": h.get("effective_mass_kg"), "peak_force_kn": h.get("peak_force_kn"),
            "gyro_nm": (r.get("robot") or {}).get("gyro_moment_nm")}


def _run_entry(r: Path, depth: int = 0, label: str | None = None) -> dict:
    info = {"path": _rel(r), "name": label or r.name, "kind": r.name.split("-")[-1] if depth == 0 else "",
            "depth": depth}
    for rep in (r / "report.json", r / "best" / "report.json"):
        if rep.exists():
            try:
                info |= _headline(json.loads(rep.read_text()))
            except Exception:
                pass
            break
    m = r / "metrics.json"
    if m.exists():  # an explicit hit
        try:
            d = json.loads(m.read_text())
            info |= {"kind": "explicit hit", "peak_kn": d.get("peak_force_kn"), "impulse_ns": d.get("impulse_ns")}
        except Exception:
            pass
    return info


def _children(r: Path) -> list[dict]:
    """Runs inside a run: topology designs (moi*/runs/*-eval, moi*/impact) and explicit-hit cases."""
    out = []
    for sub in sorted(p for p in r.iterdir() if p.is_dir()):
        if sub.name in ("fea", "vtk", "anim", "best"):
            continue
        if (sub / "metrics.json").exists() or (sub / "vtk").exists():
            out.append(_run_entry(sub, 1, sub.name.replace("_", " ")))
        if (sub / "eval" / "report.json").exists():  # a topology design evaluated by the sweep
            out.append(_run_entry(sub / "eval", 1, f"{sub.name} · eval"))
        for ev in sorted((sub / "runs").glob("*")) if (sub / "runs").exists() else []:
            if ev.is_dir():
                out.append(_run_entry(ev, 1, f"{sub.name} · {ev.name.split('-')[-1]} {ev.name[9:11]}:{ev.name[11:13]}"))
        for case in sorted((sub / "impact").glob("*/*")) if (sub / "impact").exists() else []:
            if case.is_dir() and (case / "metrics.json").exists():
                out.append(_run_entry(case, 1, f"{sub.name} · explicit {case.name}"))
    return out


def project(name: str) -> dict:
    d = PROJECTS / name
    runs = []
    for r in sorted((d / "runs").glob("*"), reverse=True) if (d / "runs").exists() else []:
        if r.is_dir():
            runs.append(_run_entry(r))
            runs += _children(r)
    return {"name": name, "spec": (d / "spec.toml").read_text(), "runs": runs}


def _anim_sources(run: Path) -> list[dict]:
    """Every animatable result under a run, in a stable order (converting one doesn't move it): each entry is
    either ready (its bundle exists) or a source to convert on first view."""
    found = []  # (sort key, title, owner dir, bundle dir, source)
    for fea in run.glob("**/fea/**/impact.frd"):
        owner = _fea_report_dir(fea.parent)
        found.append((_rel(owner) + "/0", "Static FEA · spin + hit to first yield", owner, _fea_anim_dir(fea.parent),
                       f"fea:{_rel(fea.parent)}"))
    for v in run.glob("**/vtk/*.pvd"):
        case = v.parent.parent
        found.append((_rel(case) + "/1", _explicit_title(case.name), case.parent, case / "anim" / "explicit",
                       f"explicit:{_rel(v.parent)}"))
    for t in run.glob("**/density_history.npz"):
        found.append((_rel(t.parent) + "/2", "Topology iterations", t.parent, t.parent / "anim" / "topo",
                      f"topo:{_rel(t)}"))
    out = []
    for _, title, owner, bundle, source in sorted(found, key=lambda x: x[0]):
        where = _rel(owner).split("/runs/")[-1]
        ready = (bundle / "manifest.json").exists()
        out.append({"source": "ready" if ready else source, "anim": _rel(bundle) if ready else None,
                    "label": f"{title} · {where}"})
    return out


def _walk(r: Path):
    """Files under a run, skipping solver/animation folders (os.walk: rglob is slow on /mnt/c)."""
    for dp, dns, fns in os.walk(r):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            yield Path(dp) / f


def _project_of(r: Path) -> str:
    return r.resolve().relative_to(PROJECTS.resolve()).parts[0]


def _latest_eval(proj_dir: Path) -> Path | None:
    for r in sorted((proj_dir / "runs").glob("*-eval"), reverse=True):
        if (r / "report.json").exists():
            return r
    return None


def catalog() -> list[dict]:
    out = []
    for p in projects():
        for r in project(p["name"])["runs"]:
            out.append(r | {"project": p["name"]})
    return out


def _explicit_title(case_dir: str) -> str:
    """design_bite_steel -> "Explicit hit · bite · steel" (reference_* -> "Reference hit · ...")."""
    who, _, rest = case_dir.partition("_")
    return f"{'Reference hit' if who == 'reference' else 'Explicit hit'} · {' · '.join(rest.split('_'))}"


def run_info(rel: str) -> dict:
    r = _safe(rel)
    info = {"path": rel, "name": r.name, "project": _project_of(r), "files": [], "images": [], "markdown": {},
            "anims": _anim_sources(r), "steps": []}
    for f in sorted(_walk(r)):
        if f.is_file():
            info["files"].append(_rel(f))
            if f.suffix == ".step":
                info["steps"].append({"path": _rel(f), "label": _rel(f.parent).split("/runs/")[-1]})
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
    # the targets the history chart draws: the mass budget and the reference's energy
    try:
        spec_path = next(pp for pp in [r, *r.parents] if (pp / "spec.toml").exists()) / "spec.toml"
        t = tomllib.loads(spec_path.read_text())
        info["targets"] = {"mass_max_g": ((t.get("constraints") or {}).get("mass_kg") or {}).get("max")}
        if info["targets"]["mass_max_g"]:
            info["targets"]["mass_max_g"] *= 1e3
        ref = (t.get("reference") or {}).get("spec")
        if ref:
            from eitri.reference import reference_metrics

            cache = (ROOT / ref).parent / ".eitri_reference.json"
            if cache.exists():
                vals = list(json.loads(cache.read_text()).values())
                if vals:
                    info["targets"]["ref_energy_j"] = vals[-1].get("energy_j")
                    info["targets"]["ref_mass_g"] = (vals[-1].get("mass_kg") or 0) * 1e3 or None
            ref_run = _latest_eval((ROOT / ref).parent)
            if ref_run is not None:
                info["reference"] = {"name": (ROOT / ref).parent.name, "path": _rel(ref_run),
                                     "headline": _headline(json.loads((ref_run / "report.json").read_text()))}
    except Exception:
        pass
    if (r / "metrics.json").exists():  # one explicit hit
        m = json.loads((r / "metrics.json").read_text())
        info["metrics"] = {k: v for k, v in m.items() if not isinstance(v, list) or k == "max_plastic_body_at_mm"}
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
        out = _fea_anim_dir(fea_dir)
        if not (out / "manifest.json").exists():
            rep = json.loads((_fea_report_dir(fea_dir) / "report.json").read_text())
            # fea/<material>/ (eval) or fea/ directly (the evals inside a topology run: one material)
            res = next((x for x in rep["results"] if x.get("material") == fea_dir.name), rep["results"][0])
            from eitri import materials

            s = res["stress"]
            anim.from_fea(fea_dir, out, s["yield_force_kn"], s["impact_at_estimated_peak"]["force_kn"],
                          materials.get(res["material"]).yield_)
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


def build_part(step_rel: str) -> str:
    from eitri.ui import part

    step = _safe(step_rel)
    out = step.parent / "anim" / f"part_{step.stem}"
    if not (out / "manifest.json").exists() or (out / "manifest.json").stat().st_mtime < step.stat().st_mtime:
        part.build(step, out)
    return _rel(out)


def timeline(rel: str) -> dict:
    """The manifest plus what the timeline draws: solver steps per frame and time curves (cached)."""
    d = _safe(rel)
    m = json.loads((d / "manifest.json").read_text())
    cache = d / "timeline.json"
    if cache.exists() and cache.stat().st_mtime >= (d / "manifest.json").stat().st_mtime:
        return m | json.loads(cache.read_text())
    extra: dict = {"steps": None, "step_unit": None, "curves": []}
    try:
        if m["kind"] == "explicit":
            extra |= _explicit_timeline(d.parent.parent, m["times"])
        elif m["kind"] == "fea":
            extra |= {"steps": list(range(len(m["times"]))), "step_unit": "load increment"}
        elif m["kind"] == "topo":
            extra |= {"steps": [int(t) for t in m["times"]], "step_unit": "optimizer iteration"}
    except Exception as e:
        extra["error"] = f"{type(e).__name__}: {e}"
    cache.write_text(json.dumps(extra))
    return m | extra


def _explicit_timeline(case: Path, times: list[float]) -> dict:
    import re

    import numpy as np

    out: dict = {}
    # solver cycles: the engine prints "cycle time dt ..." every few hundred cycles
    eng = sorted(case.glob("*_0001.out"))
    if eng:
        rows = [(int(a), float(b)) for a, b in re.findall(r"^\s+(\d+)\s+([0-9.]+(?:E[-+]?\d+)?)\s+[0-9.]+E[-+]\d+\s",
                                                           eng[0].read_text(errors="ignore"), re.M)]
        if rows:
            cyc, t = np.array(rows).T
            out["steps"] = [int(round(v)) for v in np.interp(times, t, cyc)]
            out["step_unit"] = "solver cycle"
            out["dt_ms"] = float(np.median(np.diff(t) / np.maximum(np.diff(cyc), 1))) if len(t) > 1 else None
    th_csv = next(iter(sorted(case.glob("*T01.csv"))), None)
    if th_csv:
        from eitri.physics.explicit.impact3d import read_th

        th = read_th(th_csv)
        tm = th.get("time")
        if tm is not None and len(tm) > 1:
            icols = [k for k in th if k.lower().startswith("contact") and " var " in k]
            curves = []
            if len(icols) >= 3:
                In = np.c_[[th[k] for k in icols[:3]]].T
                fn = np.nan_to_num(np.linalg.norm(np.gradient(In, tm, axis=0), axis=1))
                curves.append({"name": "contact force", "unit": "kN", "t": tm.tolist(), "v": fn.tolist()})
            for k, name in (("PLASTIC WORK", "plastic work"), ("ROTATION ENERGY", "rotation energy"),
                            ("KINETIC ENERGY", "kinetic energy"), ("INTERNAL ENERGY", "internal energy")):
                if k in th:
                    curves.append({"name": name, "unit": "J", "t": tm.tolist(), "v": np.nan_to_num(th[k]).tolist()})
            out["curves"] = curves
    return out


def _fea_report_dir(fea_dir: Path) -> Path:
    return fea_dir.parent if fea_dir.name == "fea" else fea_dir.parent.parent


def _fea_anim_dir(fea_dir: Path) -> Path:
    return _fea_report_dir(fea_dir) / "anim" / ("fea" if fea_dir.name == "fea" else f"fea_{fea_dir.name}")


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


def _load_jobs() -> dict[int, dict]:
    try:
        return {int(k): v for k, v in json.loads(JOBS_FILE.read_text()).items()}
    except Exception:
        return {}


def _save_jobs(js: dict[int, dict]) -> None:
    JOBS_FILE.write_text(json.dumps(js, indent=1))


def start_job(cmd: str, proj: str, args: list[str], spec_rel: str | None = None) -> dict:
    """The CLI in a detached shell that records its exit code: jobs outlive a server restart (hot reload).
    `spec_rel` targets a design (a run's spec_fixed.toml) instead of the project spec."""
    import shlex

    if cmd not in COMMANDS:
        raise ValueError(cmd)
    spec = _safe(spec_rel) if spec_rel else PROJECTS / proj / "spec.toml"
    log = PROJECTS / proj / "runs" / f"job-{int(time.time())}-{cmd}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ROOT / "eitri.sh"), "eitri", cmd, str(spec.relative_to(ROOT)), *[str(a) for a in args]]
    sh = f"{shlex.join(argv)} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(log))}.rc"
    p = subprocess.Popen(["sh", "-c", sh], cwd=ROOT, start_new_session=True)
    with _job_lock:
        js = _load_jobs()
        jid = max(js, default=0) + 1
        js[jid] = {"id": jid, "cmd": cmd, "project": proj, "args": args, "log": _rel(log), "pid": p.pid,
                   "spec": _rel(spec),
                   "started": time.time()}
        _save_jobs(js)
    return {"id": jid}


def _job_state(j: dict) -> tuple[bool, int | None]:
    rc = ROOT / (j["log"] + ".rc")
    if rc.exists():
        try:
            return False, int(rc.read_text().strip() or -1)
        except ValueError:
            return False, -1
    try:
        os.kill(j["pid"], 0)
        return True, None
    except OSError:
        return False, None  # gone without an exit code (killed)


def jobs() -> list[dict]:
    out = []
    for j in _load_jobs().values():
        running, rc = _job_state(j)
        out.append(j | {"running": running, "returncode": rc})
    return sorted(out, key=lambda j: -j["id"])


def job_log(jid: int, tail: int = 20000) -> dict:
    j = _load_jobs()[jid]
    text = (ROOT / j["log"]).read_text(errors="ignore")
    noise = ("zoxide", "Please ensure", "If the issue", "github.com/ajeet", "Disable this message")
    text = "\n".join(l for l in text.splitlines() if not l.startswith(noise) and l.strip() != "")
    return {"id": jid, "running": _job_state(j)[0], "log": text[-tail:]}


# ---------- live events: static files and project data ----------
_subscribers: list[queue.Queue] = []
_sub_lock = threading.Lock()


def _publish(event: str, data: dict) -> None:
    with _sub_lock:
        for q in _subscribers:
            q.put((event, data))


def _signature_static() -> dict[str, float]:
    return {f.name: f.stat().st_mtime for f in STATIC.iterdir() if f.is_file()}


def _signature_projects() -> dict[str, float]:
    """Directory mtimes down the run trees (a new file or folder changes its parent's mtime): cheap on /mnt/c,
    unlike stat-ing every file."""
    sig = {}
    for proj in PROJECTS.iterdir():
        if not proj.is_dir() or proj.name.startswith("."):
            continue
        for dp, dns, _ in os.walk(proj):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            if dp.count(os.sep) - str(PROJECTS).count(os.sep) > 7:
                dns[:] = []
            try:
                sig[dp] = os.stat(dp).st_mtime
            except OSError:
                pass
        spec = proj / "spec.toml"
        if spec.exists():
            sig[str(spec)] = spec.stat().st_mtime
    return sig


def _watch() -> None:
    st, pr = _signature_static(), _signature_projects()
    n = 0
    while True:
        time.sleep(0.5)
        new = _signature_static()
        if new != st:
            changed = sorted(k for k in new if new.get(k) != st.get(k))
            st = new
            _publish("static", {"files": changed})
        n += 1
        if n % 6 == 0:  # projects every 3 s
            new = _signature_projects()
            if new != pr:
                keys = {k for k in set(new) | set(pr) if new.get(k) != pr.get(k)}
                names = sorted({Path(k).resolve().relative_to(PROJECTS.resolve()).parts[0] for k in keys})
                pr = new
                _publish("data", {"projects": names})


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

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q: queue.Queue = queue.Queue()
        with _sub_lock:
            _subscribers.append(q)
        try:
            self.wfile.write(f"event: hello\ndata: {json.dumps({'boot': BOOT_ID})}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev, data = q.get(timeout=15)
                    msg = f"event: {ev}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    msg = ": ping\n\n"
                self.wfile.write(msg.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sub_lock:
                _subscribers.remove(q)

    def do_GET(self):
        path, q = self._query()
        try:
            if path in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if not str(f).startswith(str(STATIC.resolve())) or not f.exists():
                    return self._json({"error": "not found"}, 404)
                ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8",
                         ".svg": "image/svg+xml"}.get(
                    f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            if path == "/api/events":
                return self._events()
            if path == "/api/projects":
                return self._json(projects())
            if path == "/api/catalog":
                return self._json(catalog())
            if path == "/api/timeline":
                return self._json(timeline(q["dir"]))
            if path == "/api/project":
                return self._json(project(q["name"]))
            if path == "/api/run":
                return self._json(run_info(q["path"]))
            if path == "/api/data":
                # Animation arrays. Never let them look like a download (no ".bin" in the URL, no octet-stream):
                # download managers (IDM) hook the browser, grab such requests, pop up and hand fetch() an
                # empty 204. A text type is never captured; fetch().arrayBuffer() reads the bytes all the same.
                f = _safe(q["dir"]) / {"mesh": "mesh.bin", "frames": "frames.bin", "vis": "vis.bin",
                                       "edges": "edges.bin", "faceid": "faceid.bin"}[q["part"]]
                return self._send(f.read_bytes(), "text/plain; charset=x-user-defined")
            if path == "/api/jobs":
                return self._json(jobs())
            if path == "/api/job":
                return self._json(job_log(int(q["id"])))
            if path.startswith("/files/"):
                f = _safe(urllib.parse.unquote(path[len("/files/"):]))
                if not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = {".png": "image/png", ".json": "application/json", ".md": "text/markdown",
                         ".toml": "text/plain", ".step": "application/step"}.get(f.suffix, "application/octet-stream")
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
            if path == "/api/part":
                return self._json({"part": build_part(q["step"])})
            if path == "/api/job":
                b = json.loads(self._body() or b"{}")
                return self._json(start_job(b["cmd"], b["project"], b.get("args", []), b.get("spec")))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_PUT(self):
        path, q = self._query()
        if path == "/api/spec":
            return self._json(save_spec(q["name"], self._body().decode()))
        return self._json({"error": "not found"}, 404)


class _Server(ThreadingHTTPServer):
    daemon_threads = True  # event streams must not keep a restarting server alive
    allow_reuse_address = True


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = _Server((host, port), Handler)
    threading.Thread(target=_watch, daemon=True).start()
    print(f"eitri ui on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.host, a.port)
