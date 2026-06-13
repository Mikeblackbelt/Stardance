"""
N-body gravitational physics engine for the solar system simulator.
Uses RK4 (Runge-Kutta 4th order) integration for accurate orbital propagation.

Units throughout:
  position     → meters (m)
  velocity     → meters/second (m/s)
  acceleration → meters/second^2 (m/s^2)
  mass         → kilograms (kg)
  time         → seconds (s)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


G = 6.674_30e-11          # gravitational constant  (m^3 kg^-1 s^-2)
AU = 1.495_978_707e11     # 1 astronomical unit     (m)
DAY = 86_400.0            # seconds per day
YEAR = 365.25 * DAY       # seconds per Julian year



@dataclass
class Body:
    name: str
    mass: float                          # kg
    pos: np.ndarray                      # [x, y, z]  m
    vel: np.ndarray                      # [vx, vy, vz] m/s
    radius: float = 1.0                  # m  (display only)
    color: tuple = (255, 255, 255)       # RGB
    is_asteroid: bool = False
    designation: Optional[str] = None   # CNEOS/MPC designation
    trail: list = field(default_factory=list)
    trail_max: int = 300

    def record_trail(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)



def _keplerian_to_cartesian(a_au, e, i_deg, Om_deg, w_deg, M0_deg, mu):
    """
    Convert Keplerian elements to heliocentric Cartesian state vectors.
    a_au  : semi-major axis (AU)
    e     : eccentricity
    i_deg : inclination (deg)
    Om_deg: longitude of ascending node (deg)
    w_deg : argument of perihelion (deg)
    M0_deg: mean anomaly at epoch (deg)
    mu    : GM of central body (m^3/s^2)
    Returns pos (m), vel (m/s) as numpy arrays [x,y,z].
    """
    a = a_au * AU
    i  = np.radians(i_deg)
    Om = np.radians(Om_deg)
    w  = np.radians(w_deg)
    M  = np.radians(M0_deg)

    # Solve Kepler's equation  M = E - e*sin(E)  via Newton iteration
    E = M
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    # True anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0)
    )

    # Distance
    r = a * (1.0 - e * np.cos(E))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Velocity in orbital plane
    p = a * (1.0 - e ** 2)
    h = np.sqrt(mu * p)
    vx_orb = -(mu / h) * np.sin(nu)
    vy_orb =  (mu / h) * (e + np.cos(nu))

    # Rotation matrices: Rz(-Om) * Rx(-i) * Rz(-w)
    cos_Om, sin_Om = np.cos(Om), np.sin(Om)
    cos_i,  sin_i  = np.cos(i),  np.sin(i)
    cos_w,  sin_w  = np.cos(w),  np.sin(w)

    # Row vectors of the rotation matrix
    Px = cos_Om * cos_w - sin_Om * sin_w * cos_i
    Py = sin_Om * cos_w + cos_Om * sin_w * cos_i
    Pz = sin_w * sin_i

    Qx = -cos_Om * sin_w - sin_Om * cos_w * cos_i
    Qy = -sin_Om * sin_w + cos_Om * cos_w * cos_i
    Qz =  cos_w * sin_i

    pos = np.array([
        Px * x_orb + Qx * y_orb,
        Py * x_orb + Qy * y_orb,
        Pz * x_orb + Qz * y_orb,
    ])
    vel = np.array([
        Px * vx_orb + Qx * vy_orb,
        Py * vx_orb + Qy * vy_orb,
        Pz * vx_orb + Qz * vy_orb,
    ])
    return pos, vel


def build_solar_system() -> List[Body]:
    """
    Return a list of Body objects for the Sun + 8 planets.
    Orbital elements from NASA JPL Fact Sheets (J2000.0 epoch).
    """
    GM_sun = G * 1.989e30  # m^3 s^-2

    # (name, mass_kg, radius_m, color,  a_AU,  e,     i,     Om,    w,     M0)
    planet_data = [
        ("Mercury", 3.301e23,  2.440e6,  (169,169,169), 0.387098, 0.205630,  7.005,  48.331,  29.124,  174.796),
        ("Venus",   4.868e24,  6.052e6,  (255,198, 77), 0.723332, 0.006772,  3.395,  76.680, 131.533,   50.115),
        ("Earth",   5.972e24,  6.371e6,  ( 70,130,180), 1.000000, 0.016708,  0.000, 174.873, 288.064,  357.517),
        ("Mars",    6.417e23,  3.390e6,  (188, 74, 60), 1.523679, 0.093401,  1.850,  49.558, 286.502,  19.373),
        ("Jupiter",  1.899e27, 7.149e7,  (255,200,130), 5.202887, 0.048498,  1.303, 100.464, 273.867,  20.020),
        ("Saturn",   5.685e26, 6.027e7,  (210,180,140), 9.536676, 0.053862,  2.489, 113.665, 339.392, 317.020),
        ("Uranus",   8.682e25, 2.556e7,  (173,216,230),19.189165, 0.047168,  0.773,  74.006,  96.998, 141.050),
        ("Neptune",  1.024e26, 2.476e7,  ( 63, 84,186),30.069923, 0.008590,  1.770, 131.784, 273.187, 256.228),
    ]

    bodies: List[Body] = []

    # Sun at origin, at rest (we work in heliocentric frame)
    bodies.append(Body(
        name="Sun",
        mass=1.989e30,
        pos=np.zeros(3),
        vel=np.zeros(3),
        radius=6.957e8,
        color=(255, 255, 100),
    ))

    for (name, mass, radius, color, a, e, inc, Om, w, M0) in planet_data:
        pos, vel = _keplerian_to_cartesian(a, e, inc, Om, w, M0, GM_sun)
        bodies.append(Body(
            name=name, mass=mass, radius=radius, color=color,
            pos=pos, vel=vel,
        ))

    return bodies


def _accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Compute gravitational accelerations for all N bodies.
    positions : (N, 3) array
    masses    : (N,)   array
    Returns   : (N, 3) acceleration array
    """
    N = len(masses)
    acc = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            r2    = np.dot(r_vec, r_vec)
            r3    = r2 * np.sqrt(r2) + 1e-10   # softening prevents div/0
            fac   = G / r3
            acc[i] += fac * masses[j] * r_vec
            acc[j] -= fac * masses[i] * r_vec
    return acc


def rk4_step(bodies: List[Body], dt: float):
    """
    Advance all bodies by one RK4 timestep dt (seconds).
    Mutates body.pos and body.vel in place.
    """
    N = len(bodies)
    masses = np.array([b.mass for b in bodies])
    pos0 = np.array([b.pos for b in bodies])
    vel0 = np.array([b.vel for b in bodies])

    def deriv(pos, vel):
        return vel, _accelerations(pos, masses)

    # k1
    dv1, da1 = deriv(pos0, vel0)
    # k2
    dv2, da2 = deriv(pos0 + 0.5*dt*dv1, vel0 + 0.5*dt*da1)
    # k3
    dv3, da3 = deriv(pos0 + 0.5*dt*dv2, vel0 + 0.5*dt*da2)
    # k4
    dv4, da4 = deriv(pos0 + dt*dv3, vel0 + dt*da3)

    new_pos = pos0 + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    new_vel = vel0 + (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)

    for i, body in enumerate(bodies):
        body.pos = new_pos[i]
        body.vel = new_vel[i]
        body.record_trail()