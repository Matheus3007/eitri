"""Weapon drive: top speed and spin-up from either a fixed rpm or a motor + reduction.

Motor model: brushless DC with linear torque-speed line, optional current limit.
    Kt = 60 / (2*pi*Kv)            [N*m/A]
    I(w_m) = min(I_lim, (V - w_m/Kv_rad) / R)
    torque at weapon = ratio * Kt * I
Top speed is the ideal no-load speed scaled by `speed_efficiency` (drag, ESC, battery sag).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp

from eitri.spec import Drive


@dataclass
class DriveResult:
    top_rpm: float
    top_rad_s: float
    spinup_s_90: float | None  # time to 90% of top speed (None if rpm given directly)
    spinup_s_95: float | None
    spinup_curve: list[tuple[float, float]] | None  # (t s, weapon rpm)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("spinup_curve")
        return d


def rpm_to_rad(rpm: float) -> float:
    return rpm * 2 * math.pi / 60


def top_speed_rad_s(drive: Drive) -> float:
    """Weapon top speed (independent of inertia): fixed rpm, or loaded motor speed / reduction."""
    if drive.rpm is not None:
        return rpm_to_rad(drive.rpm)
    m = drive.motor
    return m.kv * 2 * math.pi / 60 * m.voltage * m.speed_efficiency / drive.ratio


def simulate(drive: Drive, weapon_moi_kgm2: float) -> DriveResult:
    if drive.rpm is not None:
        return DriveResult(drive.rpm, rpm_to_rad(drive.rpm), None, None, None)

    m = drive.motor
    kv_rad = m.kv * 2 * math.pi / 60  # rad/s per V
    kt = 1 / kv_rad
    w_free_motor = kv_rad * m.voltage
    w_top = w_free_motor * m.speed_efficiency / drive.ratio  # weapon rad/s
    j = weapon_moi_kgm2 + m.rotor_inertia_kgm2 * drive.ratio**2

    def torque(w_weapon: float) -> float:
        w_m = w_weapon * drive.ratio
        if m.resistance_ohm > 0:
            i = (m.voltage - w_m / kv_rad) / m.resistance_ohm
        else:
            i = math.inf
        if m.current_limit_a is not None:
            i = min(i, m.current_limit_a)
        if not math.isfinite(i):
            raise ValueError("motor needs resistance_ohm or current_limit_a to bound torque")
        return drive.ratio * kt * max(i, 0.0)

    def rhs(_t, y):
        # Stop accelerating at the loaded top speed (efficiency lumps the losses we don't model).
        return [torque(y[0]) / j if y[0] < w_top else 0.0]

    def reach(frac):
        ev = lambda _t, y: y[0] - frac * w_top  # noqa: E731
        ev.terminal = frac == 0.95
        ev.direction = 1
        return ev

    sol = solve_ivp(rhs, (0, 120), [0.0], events=[reach(0.90), reach(0.95)],
                    max_step=0.01, rtol=1e-8, atol=1e-6)
    t90 = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    t95 = float(sol.t_events[1][0]) if len(sol.t_events[1]) else None
    idx = np.linspace(0, len(sol.t) - 1, min(len(sol.t), 200)).astype(int)
    curve = [(float(sol.t[i]), float(sol.y[0][i] * 60 / (2 * math.pi))) for i in idx]
    return DriveResult(w_top * 60 / (2 * math.pi), w_top, t90, t95, curve)
