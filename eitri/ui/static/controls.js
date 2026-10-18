// CAD navigation, SolidWorks style. Rotation grabs the part: it turns the way the cursor moves (screen-space
// trackball about the pivot, no fixed up axis). Pan keeps the point under the cursor under the cursor. The wheel
// zooms about the cursor, SolidWorks' direction (wheel forward = zoom out; flip ZOOM_FORWARD_OUT for the other).
// Buttons: left or middle drag rotate; right drag, Ctrl/Shift + middle, or Ctrl + left pan; double-click fits.
// Keys (while the pointer is over the view): F fit, Ctrl+1 front, Ctrl+3 left, Ctrl+5 top, Ctrl+7 isometric.
import * as THREE from "three";

const ZOOM_FORWARD_OUT = true;
const ROT_SPEED = 0.0065;  // rad per pixel
const ease = (t) => 1 - Math.pow(1 - t, 3);

export class CadControls extends THREE.EventDispatcher {
  constructor(camera, dom, { pick = null, onFit = null, onView = null } = {}) {
    super();
    Object.assign(this, { camera, dom, pick, onFit, onView });
    this.target = new THREE.Vector3();
    this.enableZoom = true;
    this.zoomGoal = 0;  // pending log-zoom, eased in over a few frames
    this.anim = null;
    dom.style.touchAction = "none";
    dom.addEventListener("contextmenu", (e) => e.preventDefault());
    dom.addEventListener("pointerdown", (e) => this.down(e));
    dom.addEventListener("pointermove", (e) => this.move(e));
    dom.addEventListener("pointerup", (e) => this.up(e));
    dom.addEventListener("pointercancel", (e) => this.up(e));
    dom.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    dom.addEventListener("dblclick", () => this.onFit?.());
    dom.addEventListener("pointerenter", () => { this.hover = true; });
    dom.addEventListener("pointerleave", () => { this.hover = false; });
    this.keys = (e) => {
      if (!this.hover || e.target.closest?.("input,textarea")) return;
      if (e.key === "f" || e.key === "F") { e.preventDefault(); this.onFit?.(); }
      const views = { 1: "front", 3: "side", 5: "top", 7: "iso" };
      if ((e.ctrlKey || e.metaKey) && views[e.key]) { e.preventDefault(); this.onView?.(views[e.key]); }
    };
    window.addEventListener("keydown", this.keys);
  }
  dispose() { window.removeEventListener("keydown", this.keys); }

  down(e) {
    const pan = e.button === 2 || (e.button === 1 && (e.ctrlKey || e.shiftKey)) || (e.button === 0 && e.ctrlKey);
    const rot = !pan && (e.button === 0 || e.button === 1);
    if (!pan && !rot) return;
    if (e.button === 1) e.preventDefault();
    this.anim = null;
    this.mode = pan ? "pan" : "rotate";
    this.last = [e.clientX, e.clientY];
    this.dom.setPointerCapture(e.pointerId);
    this.dom.style.cursor = pan ? "move" : "grabbing";
    this.dispatchEvent({ type: "start" });
  }
  move(e) {
    if (!this.mode) return;
    const dx = e.clientX - this.last[0], dy = e.clientY - this.last[1];
    this.last = [e.clientX, e.clientY];
    if (this.mode === "rotate") this.rotate(dx, dy); else this.pan(dx, dy);
    this.dispatchEvent({ type: "change" });
  }
  up(e) {
    if (!this.mode) return;
    this.mode = null;
    this.dom.style.cursor = "";
    try { this.dom.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    this.dispatchEvent({ type: "end" });
  }
  rotate(dx, dy) {
    // the model turns with the cursor: about the camera's up axis for dx, its right axis for dy;
    // the camera does the inverse about the pivot
    const cam = this.camera;
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
    const q = new THREE.Quaternion().setFromAxisAngle(up, dx * ROT_SPEED)
      .multiply(new THREE.Quaternion().setFromAxisAngle(right, dy * ROT_SPEED)).invert();
    const off = cam.position.clone().sub(this.target).applyQuaternion(q);
    cam.up.copy(up.applyQuaternion(q));
    cam.position.copy(this.target).add(off);
    cam.lookAt(this.target);
    cam.updateMatrixWorld();
  }
  worldPerPixel() {
    const d = this.camera.position.distanceTo(this.target);
    return 2 * d * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) / (this.dom.clientHeight || 1);
  }
  pan(dx, dy) {
    const cam = this.camera, s = this.worldPerPixel();
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
    const v = right.multiplyScalar(-dx * s).add(up.multiplyScalar(dy * s));
    cam.position.add(v); this.target.add(v);
    cam.updateMatrixWorld();
  }
  wheel(e) {
    if (!this.enableZoom) return;
    e.preventDefault();
    this.anim = null;
    const r = this.dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    // zoom about the surface under the cursor, else the point on the pivot's depth plane under it
    let p = this.pick?.(ndc);
    if (!p) {
      const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, this.camera);
      const n = new THREE.Vector3(); this.camera.getWorldDirection(n);
      p = new THREE.Vector3();
      if (!rc.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, this.target), p)) p = this.target.clone();
    }
    this.zoomAt = p;
    const dir = Math.sign(e.deltaY) * (ZOOM_FORWARD_OUT ? -1 : 1);  // deltaY < 0 = wheel forward
    this.zoomGoal += dir * Math.min(0.24, Math.abs(e.deltaY) / 900 + 0.07);  // ~19 % per notch
  }
  applyZoom(z) {
    const s = Math.exp(z), p = this.zoomAt || this.target, cam = this.camera;
    const d = cam.position.distanceTo(p);
    if ((d * s < cam.near * 4 && s < 1) || (d * s > cam.far / 4 && s > 1)) return;
    cam.position.sub(p).multiplyScalar(s).add(p);
    this.target.sub(p).multiplyScalar(s).add(p);
    cam.updateMatrixWorld();
  }
  // a smooth camera move (views, fit): position, pivot and up eased together
  animateTo(position, target, up, ms = 520) {
    const cam = this.camera;
    const from = { p: cam.position.clone(), t: this.target.clone(), q: cam.quaternion.clone() };
    const m = new THREE.Matrix4().lookAt(position, target, up);
    const qTo = new THREE.Quaternion().setFromRotationMatrix(m);
    this.anim = { from, to: { p: position.clone(), t: target.clone(), q: qTo, up: up.clone() }, t0: performance.now(), ms };
  }
  update() {
    const cam = this.camera;
    if (Math.abs(this.zoomGoal) > 1e-4) {
      const step = this.zoomGoal * 0.28;
      this.zoomGoal -= step;
      this.applyZoom(step);
      this.dispatchEvent({ type: "change" });
    }
    if (this.anim) {
      const a = this.anim, u = Math.min(1, (performance.now() - a.t0) / a.ms), k = ease(u);
      // orbit-like path: interpolate the offset direction on the sphere, not the straight line
      const r0 = a.from.p.clone().sub(a.from.t), r1 = a.to.p.clone().sub(a.to.t);
      const len = THREE.MathUtils.lerp(r0.length(), r1.length(), k);
      const qd = new THREE.Quaternion().setFromUnitVectors(r0.clone().normalize(), r1.clone().normalize());
      const dir = r0.clone().normalize().applyQuaternion(new THREE.Quaternion().slerp(qd, k));
      this.target.lerpVectors(a.from.t, a.to.t, k);
      cam.position.copy(this.target).addScaledVector(dir, len);
      cam.quaternion.slerpQuaternions(a.from.q, a.to.q, k);
      cam.up.set(0, 1, 0).applyQuaternion(cam.quaternion);
      cam.updateMatrixWorld();
      if (u >= 1) { cam.up.copy(a.to.up); cam.lookAt(this.target); this.anim = null; }
      this.dispatchEvent({ type: "change" });
    }
  }
}
