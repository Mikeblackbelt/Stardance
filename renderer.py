"""
3D Solar System Renderer - Fast + Accurate Visual Scaling
"""

import pygame
import numpy as np
from typing import List, Tuple
from physics import Body, AU

BG_COLOR = (3, 3, 12)
GRID_COLOR = (25, 25, 45)
HUD_COLOR = (210, 210, 210)
HUD_SHADOW = (0, 0, 0)
TRAIL_ALPHA = 140
CLOSE_APPROACH_COLOR = (255, 70, 70)

class Camera:
    def __init__(self):
        self.distance = 8.0 * AU
        self.yaw = 25.0
        self.pitch = 30.0
        self.target = np.zeros(3)

    def get_view_matrix(self):
        theta = np.radians(self.yaw)
        phi = np.radians(self.pitch)
        x = self.distance * np.cos(phi) * np.sin(theta)
        y = self.distance * np.cos(phi) * np.cos(theta)
        z = self.distance * np.sin(phi)
        eye = np.array([x, y, z]) + self.target

        f = eye - self.target
        f /= np.linalg.norm(f)
        s = np.cross(f, np.array([0., 0., 1.]))
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        view = np.eye(4)
        view[:3, 0] = s
        view[:3, 1] = u
        view[:3, 2] = -f
        view[:3, 3] = -np.dot([s, u, -f], eye)
        return view

    def get_projection_matrix(self, width, height, fov=52.0):
        aspect = width / height
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2)
        near, far = 5e7, 3e13
        proj = np.zeros((4, 4))
        proj[0,0] = f / aspect
        proj[1,1] = f
        proj[2,2] = (far + near) / (near - far)
        proj[2,3] = 2 * far * near / (near - far)
        proj[3,2] = -1
        return proj


class SolarSystemRenderer:
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        pygame.display.set_caption("Solar System Simulator — 3D NASA NEO")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 17, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 13)

        self.width = width
        self.height = height
        self.camera = Camera()

        self.dragging = False
        self.right_dragging = False

        self.paused = False
        self.time_multiplier = 1.0
        self.sim_dt = 3600.0
        self.show_hud = True
        self.show_asteroid_trails = True
        self.show_planet_trails = True
        self.elapsed_days = 0.0
        self.close_approach_au = 0.05
        self.close_view = False  # Press V to toggle

        self.trail_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        self.grid_surf = None

    def _project(self, pos: np.ndarray) -> Tuple[int, int, float]:
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(self.width, self.height)
        point = np.append(pos, 1.0)
        clip = proj @ view @ point
        if abs(clip[3]) < 1e-9:
            return -9999, -9999, -9999
        ndc = clip[:3] / clip[3]
        x = int((ndc[0] * 0.5 + 0.5) * self.width)
        y = int((1.0 - (ndc[1] * 0.5 + 0.5)) * self.height)
        return x, y, ndc[2]

    def _draw_grid(self):
        if self.grid_surf is None or self.grid_surf.get_size() != (self.width, self.height):
            self.grid_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.grid_surf.fill((0,0,0,0))
            max_r = 35 if not self.close_view else 8
            for r_au in range(1, max_r):
                r = r_au * AU
                points = []
                steps = 48 if r_au > 10 else 64
                for i in range(steps):
                    ang = i * 2 * np.pi / steps
                    p = np.array([r * np.cos(ang), r * np.sin(ang), 0.0])
                    sx, sy, _ = self._project(p)
                    if 0 <= sx <= self.width and 0 <= sy <= self.height:
                        points.append((sx, sy))
                if len(points) > 1:
                    pygame.draw.lines(self.grid_surf, GRID_COLOR, False, points, 1)
        self.screen.blit(self.grid_surf, (0, 0))

    def _body_screen_radius(self, body: Body, depth: float) -> int:
        dist = max(1e10, abs(depth))

        if body.name == "Sun":
            return max(12, int(22 * (self.camera.distance / dist) ** 0.35))

        if body.is_asteroid:
            return 3   # asteroids always small

        # Planets: real size + strong log exaggeration for visibility
        real_factor = body.radius / 1e6
        log_scale = 4.0 + np.log10(real_factor) * 1.8
        dist_factor = (self.camera.distance / dist) ** 0.45

        size = max(3, int(log_scale * dist_factor))
        return min(size, 32)

    def _draw_trail(self, body: Body):
        if len(body.trail) < 3:
            return
        if body.is_asteroid and not self.show_asteroid_trails:
            return
        if not body.is_asteroid and body.name != "Sun" and not self.show_planet_trails:
            return

        # Heavy decimation for performance
        step = max(1, len(body.trail) // 140)
        projected = []
        for p in body.trail[::step]:
            x, y, _ = self._project(p)
            if 0 <= x <= self.width and 0 <= y <= self.height:
                projected.append((x, y))

        for i in range(1, len(projected)):
            alpha = int(TRAIL_ALPHA * i / len(projected))
            color = (*body.color, alpha)
            pygame.draw.line(self.trail_surf, color, projected[i-1], projected[i], 1)

    def _draw_body(self, body: Body):
        sx, sy, depth = self._project(body.pos)
        if not (0 <= sx <= self.width and 0 <= sy <= self.height):
            return

        r = self._body_screen_radius(body, depth)

        if body.name == "Sun":
            glow = pygame.Surface((r*10, r*10), pygame.SRCALPHA)
            for i in range(r*5, 0, -2):
                a = max(12, int(95 * (1 - i / (r*5))))
                pygame.draw.circle(glow, (255, 245, 100, a), (r*5, r*5), i)
            self.screen.blit(glow, (sx - r*5, sy - r*5))

        if body.is_asteroid:
            r_au = np.linalg.norm(body.pos) / AU
            if r_au < self.close_approach_au:
                pygame.draw.circle(self.screen, CLOSE_APPROACH_COLOR, (sx, sy), r + 6, 2)

        pygame.draw.circle(self.screen, body.color, (sx, sy), max(r, 2))

    def _draw_label(self, body: Body):
        if not self.show_hud:
            return
        sx, sy, _ = self._project(body.pos)
        label = body.name if not body.is_asteroid else (body.designation or body.name[:10])
        surf = self.font_small.render(label, True, body.color)
        shadow = self.font_small.render(label, True, HUD_SHADOW)
        self.screen.blit(shadow, (sx + 11, sy - 8))
        self.screen.blit(surf,   (sx + 10, sy - 9))

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.time_multiplier = min(self.time_multiplier * 2, 2048)
                elif event.key == pygame.K_MINUS:
                    self.time_multiplier = max(self.time_multiplier / 2, 0.03125)
                elif event.key == pygame.K_a:
                    self.show_asteroid_trails = not self.show_asteroid_trails
                elif event.key == pygame.K_p:
                    self.show_planet_trails = not self.show_planet_trails
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key == pygame.K_r:
                    self.camera = Camera()
                elif event.key == pygame.K_v:          # New: toggle close view
                    self.close_view = not self.close_view
                    if self.close_view:
                        self.camera.distance = 2.5 * AU
                    else:
                        self.camera.distance = 8.0 * AU
                    self.grid_surf = None

            # Mouse controls...
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: self.dragging = True
                elif event.button == 3: self.right_dragging = True
                elif event.button == 4: self.camera.distance /= 1.18
                elif event.button == 5: self.camera.distance *= 1.18

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: self.dragging = False
                elif event.button == 3: self.right_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                if self.dragging:
                    self.camera.yaw += dx * 0.45
                    self.camera.pitch = np.clip(self.camera.pitch - dy * 0.45, -88, 88)
                if self.right_dragging:
                    speed = self.camera.distance * 0.00055
                    self.camera.target[0] -= dx * speed
                    self.camera.target[1] += dy * speed

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.trail_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                self.grid_surf = None

        # WASD
        keys = pygame.key.get_pressed()
        move = self.camera.distance * 0.001
        if keys[pygame.K_w]: self.camera.target[2] += move
        if keys[pygame.K_s]: self.camera.target[2] -= move
        if keys[pygame.K_a]: self.camera.target[0] -= move
        if keys[pygame.K_d]: self.camera.target[0] += move
        if keys[pygame.K_q]: self.camera.distance *= 1.025
        if keys[pygame.K_e]: self.camera.distance /= 1.025

        return True

    def render(self, bodies: List[Body]):
        self.screen.fill(BG_COLOR)
        self._draw_grid()

        fade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        fade.fill((0, 0, 0, 24))
        self.trail_surf.blit(fade, (0, 0))

        for body in bodies:
            self._draw_trail(body)
        self.screen.blit(self.trail_surf, (0, 0))

        # Depth sort
        sorted_bodies = sorted(bodies, key=lambda b: -np.linalg.norm(b.pos))
        for body in sorted_bodies:
            self._draw_body(body)

        for body in [b for b in bodies if b.name != "Sun"]:
            self._draw_label(body)

        fps = self.clock.get_fps()
        self._draw_hud(bodies, fps)

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_hud(self, bodies, fps):
        if not self.show_hud: return
        lines = [
            f"FPS: {fps:5.1f}   Day: {self.elapsed_days:,.0f}",
            f"Speed: {self.time_multiplier:.1f}×   Cam: {self.camera.distance/AU:.2f} AU",
            "V = Toggle Close/System View",
        ]

        close = sum(1 for b in bodies if b.is_asteroid and np.linalg.norm(b.pos)/AU < self.close_approach_au)
        if close:
            lines.insert(2, f"⚠ CLOSE APPROACHES: {close}")

        y = 12
        for line in lines:
            col = CLOSE_APPROACH_COLOR if line.startswith("⚠") else HUD_COLOR
            surf = self.font_small.render(line, True, col)
            self.screen.blit(surf, (14, y))
            y += 19

        state = "PAUSED" if self.paused else "RUNNING"
        status = self.font_large.render(state, True, (255,80,80) if self.paused else (80,255,80))
        self.screen.blit(status, (self.width - 170, 12))

    def quit(self):
        pygame.quit()