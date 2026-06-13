"""
Usage:
  python main.py [--no-asteroids] [--max-asteroids N] [--dt-hours H]

Flags:
  --no-asteroids      Skip NASA API fetch (offline mode, planets only)
  --max-asteroids N   Fetch at most N asteroids (default 10)
  --dt-hours H        Physics timestep in hours (default 1.0)
  --headless-test     Run 100 steps then exit (CI smoke test)
"""

import sys
import time
import argparse
import threading
import numpy as np

from physics import build_solar_system, rk4_step, DAY
from renderer import SolarSystemRenderer
from asteroid_fetcher import fetch_asteroids


def parse_args():
    p = argparse.ArgumentParser(description="Solar System Simulator with NASA NEO tracking")
    p.add_argument("--no-asteroids",   action="store_true",
                   help="Run without fetching asteroid data (offline mode)")
    p.add_argument("--max-asteroids",  type=int, default=10,
                   help="Maximum number of asteroids to fetch (default 10)")
    p.add_argument("--dt-hours",       type=float, default=1.0,
                   help="Physics timestep in hours (default 1.0)")
    p.add_argument("--headless-test",  action="store_true",
                   help="Run 100 integration steps then exit (no display)")
    return p.parse_args()



class AsteroidLoader:
    def __init__(self, max_count: int):
        self.max_count = max_count
        self.bodies    = []
        self.done      = False
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        try:
            self.bodies = fetch_asteroids(max_count=self.max_count)
        except Exception as e:
            print(f"[AsteroidLoader] Error: {e}")
        self.done = True



def headless_test(bodies, dt_s):
    print("[headless-test] Running 100 RK4 steps …")
    for step in range(100):
        rk4_step(bodies, dt_s)
    earth = next(b for b in bodies if b.name == "Earth")
    r_au  = np.linalg.norm(earth.pos) / 1.495978707e11
    print(f"[headless-test] Earth r after 100×{dt_s/3600:.1f}h steps = {r_au:.6f} AU")
    print("[headless-test] PASS" if 0.98 < r_au < 1.02 else "[headless-test] WARNING: drift detected")

def main():
    args = parse_args()
    dt_s = args.dt_hours * 3600.0

    print("═" * 60)
    print("  Solar System Simulator")
    print("  Physics: RK4 N-body  |  Data: NASA SSD/CNEOS + NeoWs")
    print("═" * 60)

    # Build planetary system
    bodies = build_solar_system()
    print(f"[main] Solar system built: {len(bodies)} bodies")

    # Headless test mode
    if args.headless_test:
        headless_test(bodies, dt_s)
        return

    # Start asteroid fetch in background
    loader = None
    if not args.no_asteroids:
        print(f"[main] Fetching up to {args.max_asteroids} asteroid(s) from NASA APIs …")
        loader = AsteroidLoader(args.max_asteroids).start()
    else:
        print("[main] Asteroid fetch disabled (--no-asteroids)")

    # Initialise renderer
    renderer = SolarSystemRenderer(width=1400, height=900)
    renderer.sim_dt = dt_s

    asteroids_added = False
    last_wall_time  = time.perf_counter()

    running = True
    while running:
        running = renderer.handle_events()

        # Integrate physics
        if not renderer.paused:
            wall_now  = time.perf_counter()
            wall_dt   = wall_now - last_wall_time
            last_wall_time = wall_now

            # How many seconds of simulation to advance this frame
            sim_advance = wall_dt * renderer.time_multiplier * DAY
            steps       = max(1, int(sim_advance / dt_s))

            for _ in range(steps):
                rk4_step(bodies, dt_s)
                renderer.elapsed_days += dt_s / DAY
        else:
            last_wall_time = time.perf_counter()

        # Merge asteroids when background fetch completes
        if loader and loader.done and not asteroids_added:
            for ast in loader.bodies:
                bodies.append(ast)
            asteroids_added = True
            n = len(loader.bodies)
            print(f"[main] {n} asteroid(s) added to simulation")

        renderer.render(bodies)

    renderer.quit()
    print("[main] Bye!")


if __name__ == "__main__":
    main()