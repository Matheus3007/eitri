"""Hit metrics for a spinner: stored energy, bite, impulse, and the loads the hit puts on the weapon.

Collision model: the tooth acts as a point mass m_eff = I / R^2 moving at the tip speed, hitting an
opponent of mass M at rest along the impact-face normal, with coefficient of restitution e:
    mu = m_eff * M / (m_eff + M)
    J  = (1 + e) * mu * v_tip                  impulse (N*s)
The pulse is taken as a half-sine of duration dt, so peak force F = (pi/2) * J / dt. That peak force is
the equivalent static tooth load used for stress checks. For a balanced rotor the axle (and through it
the robot) reacts the full impulse J, so F is also the peak shaft/bearing load estimate.
"""

import math
from dataclasses import asdict, dataclass

from eitri.spec import Context


@dataclass
class ImpactResult:
    energy_j: float
    tip_speed_mps: float
    effective_mass_kg: float
    bite_mm: float  # min(kinematic bite, tooth height)
    bite_kinematic_mm: float  # closing distance between successive teeth
    bite_geometric_mm: float  # tooth height available to engage
    impulse_ns: float
    peak_force_kn: float
    energy_to_opponent_j: float
    weapon_energy_lost_j: float
    weapon_speed_drop_pct: float
    inward_force_fraction: float  # sin(hook): + pulls the opponent toward the axle, - pushes it out

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(moi_kgm2: float, omega_rad_s: float, tip_radius_mm: float, n_teeth: int,
            tooth_height_mm: float, hook_deg: float, ctx: Context) -> ImpactResult:
    R = tip_radius_mm * 1e-3
    v = omega_rad_s * R
    m_eff = moi_kgm2 / R**2
    M = ctx.opponent_mass_kg

    t_between_teeth = 2 * math.pi / (n_teeth * omega_rad_s) if omega_rad_s > 0 else math.inf
    bite_kin = ctx.closing_speed_mps * t_between_teeth * 1e3

    mu = m_eff * M / (m_eff + M)
    J = (1 + ctx.restitution) * mu * v
    F_peak = math.pi / 2 * J / (ctx.contact_time_ms * 1e-3)
    v_after = v - J / m_eff
    e_lost = 0.5 * m_eff * (v**2 - v_after**2)

    return ImpactResult(
        energy_j=0.5 * moi_kgm2 * omega_rad_s**2,
        tip_speed_mps=v,
        effective_mass_kg=m_eff,
        bite_mm=min(bite_kin, tooth_height_mm),
        bite_kinematic_mm=bite_kin,
        bite_geometric_mm=tooth_height_mm,
        impulse_ns=J,
        peak_force_kn=F_peak * 1e-3,
        energy_to_opponent_j=J**2 / (2 * M),
        weapon_energy_lost_j=e_lost,
        weapon_speed_drop_pct=100 * (1 - v_after / v) if v > 0 else 0.0,
        inward_force_fraction=math.sin(math.radians(hook_deg)),
    )
