"""What the weapon does to the rest of the robot, and how much damage it tolerates. Closed forms.

gyro       A rotor spinning at omega whose axis is turned at rate Omega needs a moment I * omega * Omega: the
           bearings and frame react it as a couple (horizontal spinner: when the robot pitches/rolls, e.g. on
           a hit or over a bump; vertical spinner: whenever it turns).
shaft      The peak hit force goes through the shaft into the bearings. Weapon at `weapon_offset_mm` from bearing
           A on a span `bearing_span_mm` (simply supported; offset < 0 = overhung outside A): bending moment at
           the weapon, bending stress 32 M / (pi d^3), and both bearing loads, gyro couple included.
fracture   Toughness from the Charpy V-notch energy (Roberts-Newton lower bound, K_IC = 8.47 CVN^0.63; Barsom-
           Rolfe's upper-shelf formula reads ~1.6x high on hardened plate), and the deepest
           surface crack (edge crack, Y = 1.12) the hot spot tolerates under the estimated peak hit before it
           runs: a_c = (K_IC / (1.12 sigma))^2 / pi. A ranking number for materials/designs, not a life estimate.
"""

import math


def gyro_moment_nm(moi_kgm2: float, omega_rad_s: float, tilt_rate_dps: float) -> float:
    return moi_kgm2 * omega_rad_s * math.radians(tilt_rate_dps)


def shaft_loads(force_n: float, gyro_nm: float, span_mm: float, offset_mm: float, d_mm: float) -> dict:
    L, a = span_mm * 1e-3, offset_mm * 1e-3
    if 0 <= a <= L:  # between the bearings
        r_a, r_b = force_n * (L - a) / L, force_n * a / L
        m = force_n * a * (L - a) / L
    else:  # overhung by |a| outside bearing A
        o = -a if a < 0 else a - L
        r_a, r_b = force_n * (L + o) / L, force_n * o / L
        m = force_n * o
    couple = gyro_nm / L
    m_total = math.hypot(m, gyro_nm * (min(a, L - a) / L if 0 <= a <= L else 1.0))
    sigma = 32 * m_total / (math.pi * (d_mm * 1e-3) ** 3) * 1e-6
    return {"bending_moment_nm": m_total, "bending_stress_mpa": sigma,
            "bearing_a_kn": math.hypot(r_a, couple) * 1e-3, "bearing_b_kn": math.hypot(r_b, couple) * 1e-3,
            "gyro_couple_kn": couple * 1e-3}


def k_ic_mpa_sqrt_m(charpy_j: float, yield_mpa: float) -> float:
    """Roberts-Newton lower bound (MPa*sqrt(m), CVN in J). `yield_mpa` is unused by this correlation but kept
    so another one can be swapped in."""
    return 8.47 * max(charpy_j, 0.0) ** 0.63


def critical_crack_mm(k_ic: float, stress_mpa: float) -> float:
    if stress_mpa <= 0:
        return math.inf
    return (k_ic / (1.12 * stress_mpa)) ** 2 / math.pi * 1e3
