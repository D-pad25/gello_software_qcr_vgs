#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

import numpy as np
import pygame
import tyro

from gello.zmq_core.robot_node import ZMQClientRobot


# ============================================================
# xArm6 FK fallback
# ============================================================
def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def xarm6_fk_xyz(joint_positions: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if q.shape[0] < 6:
        raise ValueError(f"Need at least 6 joints for xArm6 FK, got {q.shape[0]}")
    q = q[:6]

    # From your commented DH model
    a = [0.0, 0.28948, 0.0775, 0.0, 0.0760, 0.0]
    alpha = [-np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0]
    d = [0.267, 0.0, 0.0, 0.3425, 0.0, 0.097]
    offsets = [0.0, -1.384, 1.38491, 0.0, 0.0, 0.0]

    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        T = T @ dh_transform(a[i], alpha[i], d[i], q[i] + offsets[i])

    return T[:3, 3]


# ============================================================
# CLI args
# ============================================================
@dataclass
class Args:
    hostname: str = "127.0.0.1"
    robot_port: int = 6001
    hz: float = 30.0
    fps: int = 30

    width: int = 1000
    height: int = 1100

    # Workspace bounds
    x_min: float = 0.25
    x_max: float = 0.55
    y_min: float = -0.1
    y_max: float = 0.1
    z_min: float = 0.4
    z_max: float = 0.6

    # Initial target
    target_x: float = 0.54
    target_y: float = 0.03
    target_z: float = 0.45

    trail_len_xy: int = 400
    trail_len_xz: int = 400

    point_radius: int = 8
    use_fk_fallback: bool = True
    show_text: bool = True


# ============================================================
# Robot observation helpers
# ============================================================
def get_robot_obs(client: ZMQClientRobot) -> dict:
    try:
        obs = client.get_observations()
        if isinstance(obs, dict):
            return obs
    except Exception:
        pass

    try:
        joints = client.get_joint_state()
        return {"joint_positions": np.asarray(joints, dtype=np.float64)}
    except Exception as e:
        raise RuntimeError(f"Failed to query robot state: {e}") from e


def extract_xyz_from_obs(obs: dict, use_fk_fallback: bool = True) -> np.ndarray:
    candidate_keys = [
        "ee_pos",
        "ee_position",
        "tcp_pos",
        "tcp_position",
        "cartesian_position",
        "position",
    ]

    for key in candidate_keys:
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 3:
                return arr[:3]

    pose_keys = ["ee_pose", "tcp_pose", "pose", "cartesian_pose"]
    for key in pose_keys:
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 3:
                return arr[:3]

    if use_fk_fallback and "joint_positions" in obs:
        return xarm6_fk_xyz(np.asarray(obs["joint_positions"], dtype=np.float64))

    raise KeyError("Could not find end-effector XYZ in observations.")


# ============================================================
# Drawing helpers
# ============================================================
def world_to_panel(
    u: float,
    v: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    rect: pygame.Rect,
    invert_u: bool = False,
    invert_v: bool = False,
) -> Tuple[int, int]:
    u_norm = (u - u_min) / max(u_max - u_min, 1e-9)
    v_norm = (v - v_min) / max(v_max - v_min, 1e-9)

    if invert_u:
        u_norm = 1.0 - u_norm
    if invert_v:
        v_norm = 1.0 - v_norm

    px = rect.left + int(np.clip(u_norm, 0.0, 1.0) * rect.width)
    py = rect.top + int((1.0 - np.clip(v_norm, 0.0, 1.0)) * rect.height)
    return px, py


def draw_panel(
    screen,
    rect: pygame.Rect,
    title: str,
    u_name: str,
    v_name: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    current_uv: Optional[Tuple[float, float]],
    target_uv: Optional[Tuple[float, float]],
    trail: deque,
    font,
    small_font,
    invert_u: bool = False,
    invert_v: bool = False,
):
    bg_color = (250, 250, 250)
    border_color = (0, 0, 0)
    grid_color = (220, 220, 220)
    axis_color = (150, 150, 150)
    robot_color = (40, 90, 220)
    target_color = (220, 50, 50)
    trail_color = (120, 120, 120)
    text_color = (20, 20, 20)

    pygame.draw.rect(screen, bg_color, rect)
    pygame.draw.rect(screen, border_color, rect, 2)

    # Grid
    for frac in np.linspace(0.0, 1.0, 11):
        x = rect.left + int(frac * rect.width)
        pygame.draw.line(screen, grid_color, (x, rect.top), (x, rect.bottom), 1)

    for frac in np.linspace(0.0, 1.0, 11):
        y = rect.top + int(frac * rect.height)
        pygame.draw.line(screen, grid_color, (rect.left, y), (rect.right, y), 1)

    # Zero axes if inside bounds
    if u_min < 0.0 < u_max:
        x0, _ = world_to_panel(
            0.0, v_min, u_min, u_max, v_min, v_max, rect,
            invert_u=invert_u, invert_v=invert_v
        )
        pygame.draw.line(screen, axis_color, (x0, rect.top), (x0, rect.bottom), 2)

    if v_min < 0.0 < v_max:
        _, y0 = world_to_panel(
            u_min, 0.0, u_min, u_max, v_min, v_max, rect,
            invert_u=invert_u, invert_v=invert_v
        )
        pygame.draw.line(screen, axis_color, (rect.left, y0), (rect.right, y0), 2)

    # Trail
    if len(trail) >= 2:
        pts = [
            world_to_panel(
                u, v, u_min, u_max, v_min, v_max, rect,
                invert_u=invert_u, invert_v=invert_v
            )
            for (u, v) in trail
        ]
        pygame.draw.lines(screen, trail_color, False, pts, 2)

    # Target
    if target_uv is not None:
        tx, ty = world_to_panel(
            target_uv[0], target_uv[1],
            u_min, u_max, v_min, v_max, rect,
            invert_u=invert_u, invert_v=invert_v
        )
        pygame.draw.circle(screen, target_color, (tx, ty), 9)
        pygame.draw.circle(screen, border_color, (tx, ty), 9, 2)

    # Current robot point
    if current_uv is not None:
        rx, ry = world_to_panel(
            current_uv[0], current_uv[1],
            u_min, u_max, v_min, v_max, rect,
            invert_u=invert_u, invert_v=invert_v
        )
        pygame.draw.circle(screen, robot_color, (rx, ry), 10)
        pygame.draw.circle(screen, border_color, (rx, ry), 10, 2)

    # Title
    title_surf = font.render(title, True, text_color)
    screen.blit(title_surf, (rect.left + 10, rect.top + 8))

    # Axis labels
    xlab = small_font.render(u_name, True, text_color)
    ylab = small_font.render(v_name, True, text_color)

    screen.blit(xlab, (rect.centerx - xlab.get_width() // 2, rect.bottom + 6))
    screen.blit(ylab, (rect.left - 20, rect.centery - ylab.get_height() // 2))

    lim_text = f"{u_name}: [{u_min:.2f}, {u_max:.2f}]   {v_name}: [{v_min:.2f}, {v_max:.2f}]"
    lim_surf = small_font.render(lim_text, True, text_color)
    screen.blit(lim_surf, (rect.left + 10, rect.bottom - 24))


# ============================================================
# Main
# ============================================================
def main(args: Args):
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Robot Visualization Client - XY and XZ")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 22)

    margin = 70
    gap = 80
    panel_width = args.width - 2 * margin
    panel_height = (args.height - 2 * margin - gap) // 2

    xy_rect = pygame.Rect(margin, margin, panel_width, panel_height)
    xz_rect = pygame.Rect(margin, margin + panel_height + gap, panel_width, panel_height)

    target_xyz = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    target_xy = (float(target_xyz[0]), float(target_xyz[1]))
    target_xz = (float(target_xyz[0]), float(target_xyz[2]))

    init_xyz = target_xyz.copy()
    init_xy = (float(init_xyz[0]), float(init_xyz[1]))
    init_xz = (float(init_xyz[0]), float(init_xyz[2]))

    sensor_contact_xyz = target_xyz.copy()
    sensor_contact_xy = (float(sensor_contact_xyz[0]), float(sensor_contact_xyz[1]))
    sensor_contact_xz = (float(sensor_contact_xyz[0]), float(sensor_contact_xyz[2]))

    trail_xy = deque(maxlen=args.trail_len_xy)
    trail_xz = deque(maxlen=args.trail_len_xz)

    latest_xyz = None
    latest_obs = None
    last_error = None

    poll_dt = 1.0 / max(args.hz, 1e-6)
    next_poll = time.perf_counter()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    trail_xy.clear()
                    trail_xz.clear()

                    # Start the new trail from the current position if available
                    if latest_xyz is not None:
                        trail_xy.append((float(latest_xyz[0]), float(latest_xyz[1])))
                        trail_xz.append((float(latest_xyz[0]), float(latest_xyz[2])))

        now = time.perf_counter()
        if now >= next_poll:
            next_poll = now + poll_dt
            try:
                latest_obs = get_robot_obs(robot_client)
                latest_xyz = extract_xyz_from_obs(
                    latest_obs, use_fk_fallback=args.use_fk_fallback
                )

                # Update target and init from robot observations if available
                if "target_position" in latest_obs:
                    t = np.asarray(latest_obs["target_position"], dtype=np.float64).reshape(-1)
                    target_xyz = t[:3]
                    target_xy = (float(target_xyz[0]), float(target_xyz[1]))
                    target_xz = (float(target_xyz[0]), float(target_xyz[2]))
                if "init_position" in latest_obs:
                    p = np.asarray(latest_obs["init_position"], dtype=np.float64).reshape(-1)
                    init_xyz = p[:3]
                    init_xy = (float(init_xyz[0]), float(init_xyz[1]))
                    init_xz = (float(init_xyz[0]), float(init_xyz[2]))
                if "sensor_contact_position" in latest_obs:
                    sc = np.asarray(latest_obs["sensor_contact_position"], dtype=np.float64).reshape(-1)
                    sensor_contact_xyz = sc[:3]
                    sensor_contact_xy = (float(sensor_contact_xyz[0]), float(sensor_contact_xyz[1]))
                    sensor_contact_xz = (float(sensor_contact_xyz[0]), float(sensor_contact_xyz[2]))

                trail_xy.append((float(latest_xyz[0]), float(latest_xyz[1])))
                trail_xz.append((float(latest_xyz[0]), float(latest_xyz[2])))
                last_error = None
            except Exception as e:
                last_error = str(e)

        screen.fill((235, 235, 235))

        current_xy = None if latest_xyz is None else (float(latest_xyz[0]), float(latest_xyz[1]))
        current_xz = None if latest_xyz is None else (float(latest_xyz[0]), float(latest_xyz[2]))

        # XY: negative x right, negative y up
        draw_panel(
            screen=screen,
            rect=xy_rect,
            title="XY View (-x right, -y up)",
            u_name="x",
            v_name="y",
            u_min=args.x_min,
            u_max=args.x_max,
            v_min=args.y_min,
            v_max=args.y_max,
            current_uv=current_xy,
            target_uv=target_xy,
            trail=trail_xy,
            font=font,
            small_font=small_font,
            invert_u=False,
            invert_v=False,
        )

        # XZ: negative x right, positive z up
        draw_panel(
            screen=screen,
            rect=xz_rect,
            title="XZ View (-x right, +z up)",
            u_name="x",
            v_name="z",
            u_min=args.x_min,
            u_max=args.x_max,
            v_min=args.z_min,
            v_max=args.z_max,
            current_uv=current_xz,
            target_uv=target_xz,
            trail=trail_xz,
            font=font,
            small_font=small_font,
            invert_u=False,
            invert_v=False,
        )

        # Draw init position as green dot on both panels
        init_color = (50, 200, 50)
        for rect, uv, u_min, u_max, v_min, v_max in [
            (xy_rect, init_xy, args.x_min, args.x_max, args.y_min, args.y_max),
            (xz_rect, init_xz, args.x_min, args.x_max, args.z_min, args.z_max),
        ]:
            ix, iy = world_to_panel(uv[0], uv[1], u_min, u_max, v_min, v_max, rect)
            pygame.draw.circle(screen, init_color, (ix, iy), 9)
            pygame.draw.circle(screen, (0, 0, 0), (ix, iy), 9, 2)

        # Draw sensor contact position as yellow dot on both panels
        sensor_color = (255, 220, 0)
        for rect, uv, u_min, u_max, v_min, v_max in [
            (xy_rect, sensor_contact_xy, args.x_min, args.x_max, args.y_min, args.y_max),
            (xz_rect, sensor_contact_xz, args.x_min, args.x_max, args.z_min, args.z_max),
        ]:
            sx, sy = world_to_panel(uv[0], uv[1], u_min, u_max, v_min, v_max, rect)
            pygame.draw.circle(screen, sensor_color, (sx, sy), 9)
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy), 9, 2)

        if args.show_text:
            text_color = (20, 20, 20)
            lines = [
                f"Robot server: {args.hostname}:{args.robot_port}",
                f"Polling: {args.hz:.1f} Hz",
                f"Target XYZ: ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}) m",
                "Press SPACE to clear trail",
            ]
            if latest_xyz is not None:
                lines.append(
                    f"Current XYZ: ({latest_xyz[0]:.3f}, {latest_xyz[1]:.3f}, {latest_xyz[2]:.3f}) m"
                )
            if last_error is not None:
                lines.append(f"ERROR: {last_error}")

            y0 = 10
            for line in lines:
                surf = small_font.render(line, True, text_color)
                screen.blit(surf, (10, y0))
                y0 += 22

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main(tyro.cli(Args))