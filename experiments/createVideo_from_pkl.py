import pickle
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# ─── I/O helpers ────────────────────────────────────────────────────────────
def load_pkl(file_path: Path):
    with file_path.open("rb") as f:
        return pickle.load(f)

def safe_to_rgb(img):
    """Ensure HxWx3 uint8 RGB."""
    img = np.asarray(img)
    if img.ndim == 2:  # grayscale
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:  # (H,W,1)
        img = np.repeat(img, 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:  # RGBA -> RGB
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / h)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

# ─── Math helpers ───────────────────────────────────────────────────────────
def quat_to_euler_xyz(w: float, x: float, y: float, z: float):
    """Return roll, pitch, yaw (deg) from quaternion (w,x,y,z), XYZ intrinsic."""
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0*(w*y - z*x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch = np.degrees(np.arcsin(t2))

    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.degrees(np.arctan2(t3, t4))
    return float(roll), float(pitch), float(yaw)

def format_ee_pose_line(ee_pos_quat):
    """ee_pos_quat expected as [x,y,z,qx,qy,qz,qw]."""
    if ee_pos_quat is None:
        return None
    ee = np.asarray(ee_pos_quat).reshape(-1)
    if ee.size != 7:
        return None
    x, y, z, qx, qy, qz, qw = [float(v) for v in ee]
    r, p, yaw = quat_to_euler_xyz(qw, qx, qy, qz)
    return (f"EE: x={x:+.3f}, y={y:+.3f}, z={z:+.3f} m  |  "
            f"r={r:+.1f}, p={p:+.1f}, y={yaw:+.1f} deg")

# ─── Frame compositor ───────────────────────────────────────────────────────
def render_frame(wrist_img: np.ndarray,
                 base_img:  np.ndarray,
                 joint_pos7: np.ndarray,   # (7,) last = gripper state
                 control7:   np.ndarray,   # (7,) last = gripper command
                 ee_pos_quat: np.ndarray | None = None,
                 prompt_text: str = "Prompt: Pick a ripe, red tomato and drop it in the blue bucket."
                 ) -> np.ndarray:
    """
    Build a presentation-quality side-by-side RGB frame with a
    semi-transparent caption including EE pose, joints, and actions.
    """
    # Normalize images and make heights match
    wrist_img = safe_to_rgb(wrist_img)
    base_img  = safe_to_rgb(base_img)
    h = min(wrist_img.shape[0], base_img.shape[0])
    wrist_img = resize_to_height(wrist_img, h)
    base_img  = resize_to_height(base_img, h)

    
    # 1) Stack wrist + base, convert to BGR for OpenCV text drawing
    frame = np.hstack([wrist_img, base_img])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    '''
    # 2) Dark translucent banner (tall enough for 4 lines)
    header_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # 3) Build strings
    ee_line = format_ee_pose_line(ee_pos_quat)

    joints_txt = (
        "Joints:  [" + ", ".join(f"{j:+.2f}" for j in np.asarray(joint_pos7).flatten()[:6]) + "]" +
        f"  |  Gripper state: {float(np.asarray(joint_pos7).flatten()[-1]):+.2f}"
    )
    actions_txt = (
        "Actions: [" + ", ".join(f"{a:+.2f}" for a in np.asarray(control7).flatten()[:6]) + "]" +
        f"  |  Gripper command: {float(np.asarray(control7).flatten()[-1]):+.2f}"
    )

    lines = [prompt_text]
    if ee_line:
        lines.append(ee_line)
    lines.append(joints_txt)
    lines.append(actions_txt)

    # 4) Pretty text rendering
    def put_line(img, text, y):
        font_scale = 0.6
        cv2.putText(img, text, (12, y+2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    ycur = 28
    for t in lines[:4]:   # up to 4 lines fit in header_h=130
        put_line(frame, t, ycur)
        ycur += 26
    '''
    return frame

# ─── Video builder ──────────────────────────────────────────────────────────
def create_video_from_pkls(pkl_folder: str | Path,
                           output_path: Path,
                           fps: int = 20):
    pkl_folder = Path(pkl_folder)
    files = sorted(pkl_folder.glob("*.pkl"))
    if not files:
        print(f"⚠️  No .pkl files found in {pkl_folder}")
        return

    print(f"🎞️  Rendering {len(files)} frames …")
    frames = []
    missing = badlen = errors = 0

    for file in tqdm(files, desc="Frames"):
        data = load_pkl(file)

        wrist = data.get("wrist_rgb")
        base = data.get("base_rgb")
        joint_pos7 = data.get("joint_positions")   # (7,) last = gripper state
        control7 = data.get("control")             # (7,) last = gripper cmd
        ee_pq = data.get("ee_pos_quat")            # [x,y,z,qx,qy,qz,qw]

        # Skip if any critical data is missing
        if any(x is None for x in (wrist, base, joint_pos7, control7)):
            missing += 1
            continue

        try:
            joint_pos7 = np.asarray(joint_pos7, dtype=float).flatten()
            control7   = np.asarray(control7, dtype=float).flatten()
            if joint_pos7.size != 7 or control7.size != 7:
                badlen += 1
                continue

            frame = render_frame(wrist, base, joint_pos7, control7, ee_pos_quat=ee_pq)
            frames.append(frame)

        except Exception as e:
            errors += 1
            # print(f"⚠️ {file.name}: {repr(e)}")
            continue

    kept = len(frames)
    print(f"ℹ️  Kept={kept}, missing={missing}, badlen={badlen}, errors={errors}")

    if kept == 0:
        print("❌ No valid frames to render. Exiting without writing a video.")
        return

    # Write video
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"✅ Saved video to {output_path}")

# ─── Run Example ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_video_from_pkls(
        pkl_folder=r"D:\ethan\gello\1014_123911",
        output_path=Path(r"D:\ethan\ethanVid.mp4"),
        fps=30
    )
