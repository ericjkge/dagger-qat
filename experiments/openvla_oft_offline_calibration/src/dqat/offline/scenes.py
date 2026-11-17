"""Re-render recorded LIBERO states, clean or with a LIBERO-Plus perturbation.

The demo HDF5s store the MuJoCo model XML and the full simulator state of every frame, so a frame
can be re-rendered exactly (validated: clean renders match the recorded 128px images to ~1-4 gray
levels) and re-rendered under a perturbation with the *same* physical state. Images are returned in
robosuite's convention (vertically flipped), i.e. what `obs["agentview_image"]` contains in the
evaluation environment, so the upstream OFT preprocessing applies unchanged.

The camera transform and image-corruption functions are ported from LIBERO-Plus
(github.com/sylvestf/LIBERO-plus: libero/libero/envs/problems/*.py and envs/env_wrapper.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LIBERO_ASSETS = ROOT / "third_party/LIBERO/libero/libero"
DEV_LIBERO = "/home/yifengz/workspace/libero-dev/chiliocosm/"
DEV_ROBOSUITE = "/home/yifengz/workspace/robosuite-master/robosuite/"
CAMERAS = ("agentview", "robot0_eye_in_hand")


def robosuite_assets() -> Path:
    import importlib.util

    # Locate the installed package without importing it (robosuite's import pulls in OpenCV).
    return Path(importlib.util.find_spec("robosuite").submodule_search_locations[0]).resolve()


def localize(xml: str) -> str:
    xml = xml.replace(DEV_LIBERO, f"{LIBERO_ASSETS}/")
    return xml.replace(DEV_ROBOSUITE, f"{robosuite_assets()}/")


# --- LIBERO-Plus camera viewpoint (ported verbatim in behavior) -----------------------------------


def _rotation():
    from scipy.spatial.transform import Rotation

    return Rotation


def scale_distance_from_pivot(original_quat=None, original_pos=None, scale_factor=1.3):
    result = {}
    pivot_point = np.array([0, 0, 0.8])
    if original_quat is not None:
        result["new_quat"] = original_quat
    if original_pos is not None:
        new_pos = pivot_point + (np.array(original_pos) - pivot_point) * scale_factor
        result["new_pos"] = new_pos.tolist()
    return result


def rotate_around_y(original_quat=None, original_pos=None, degrees=0):
    Rotation = _rotation()
    result = {}
    axis = np.array([0, 1, 0])
    axis_point = np.array([0, 0, 0.8])
    custom_rotation = Rotation.from_rotvec(np.radians(-degrees) * axis)
    if original_quat is not None:
        q = original_quat
        new = (custom_rotation * Rotation.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
        result["new_quat"] = [float(new[3]), float(new[0]), float(new[1]), float(new[2])]
    if original_pos is not None:
        result["new_pos"] = (custom_rotation.apply(np.array(original_pos) - axis_point) + axis_point).tolist()
    return result


def rotate_around_z(original_quat=None, original_pos=None, degrees=0):
    Rotation = _rotation()
    result = {}
    z_rotation = Rotation.from_euler("z", degrees, degrees=True)
    if original_quat is not None:
        q = original_quat
        new = (z_rotation * Rotation.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
        result["new_quat"] = [float(new[3]), float(new[0]), float(new[1]), float(new[2])]
    if original_pos is not None:
        result["new_pos"] = z_rotation.apply(original_pos).tolist()
    return result


def camera_pose(pos_av, quat_av, view):
    """LIBERO-Plus `_setup_camera` for view = (horizon, vertical, scale%, end_rot, end_vertical)."""
    view_h, up_view, scale, end_rot, end_vertical = view
    r4 = lambda xs: [round(x, 4) for x in xs]  # noqa: E731 (matches upstream rounding)
    if int(up_view) != 0:
        up = rotate_around_y(original_quat=quat_av, original_pos=pos_av, degrees=int(up_view))
        res = rotate_around_z(original_quat=r4(up["new_quat"]), original_pos=r4(up["new_pos"]), degrees=int(view_h))
    else:
        res = rotate_around_z(original_quat=quat_av, original_pos=pos_av, degrees=int(view_h))
    pos, quat = r4(res["new_pos"]), r4(res["new_quat"])
    if scale / 100 != 1.0:
        res = scale_distance_from_pivot(original_quat=quat, original_pos=pos, scale_factor=scale / 100)
        pos, quat = r4(res["new_pos"]), r4(res["new_quat"])
    if int(end_rot) != 0:
        quat = r4(rotate_around_z(original_quat=quat, degrees=int(end_rot))["new_quat"])
    if int(end_vertical) != 0:
        quat = r4(rotate_around_y(original_quat=quat, degrees=int(end_vertical))["new_quat"])
    return pos, quat


def _set_attrs(xml: str, tag: str, name: str, attrs: dict) -> str:
    pattern = re.compile(rf'<{tag}\b[^>]*\bname="{re.escape(name)}"[^>]*?/?>')
    match = pattern.search(xml)
    if match is None:
        raise KeyError(f"<{tag} name={name!r}> not in model XML")
    element = match.group(0)
    for key, value in attrs.items():
        if re.search(rf'\b{key}="[^"]*"', element):
            element = re.sub(rf'\b{key}="[^"]*"', f'{key}="{value}"', element)
        else:
            element = element.replace(f"<{tag} ", f'<{tag} {key}="{value}" ', 1)
    return xml[: match.start()] + element + xml[match.end() :]


def apply_camera(xml: str, view) -> str:
    element = re.search(r'<camera\b[^>]*\bname="agentview"[^>]*>', xml).group(0)
    pos = [float(v) for v in re.search(r'pos="([^"]+)"', element).group(1).split()]
    quat = [float(v) for v in re.search(r'quat="([^"]+)"', element).group(1).split()]
    new_pos, new_quat = camera_pose(pos, quat, view)
    return _set_attrs(xml, "camera", "agentview", {"pos": " ".join(map(str, new_pos)), "quat": " ".join(map(str, new_quat))})


def apply_scene_patch(xml: str, patch: dict, asset_files: dict[str, Path]) -> tuple[str, int]:
    """Apply light/texture attribute changes (by element name) present in this model XML."""
    applied = 0
    for tag in ("light", "texture"):
        for name, attrs in patch[tag].items():
            if not re.search(rf'<{tag}\b[^>]*\bname="{re.escape(name)}"', xml):
                continue
            attrs = dict(attrs)
            if "file" in attrs:
                attrs["file"] = str(asset_files[attrs["file"]])
            xml = _set_attrs(xml, tag, name, attrs)
            applied += 1
    return xml, applied


# --- LIBERO-Plus sensor noise (gaussian blur, zoom blur, fog, glass blur) -------------------------


def gaussian_blur(x, severity=1):
    from skimage.filters import gaussian

    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][severity - 1]
    return np.clip(gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1), 0, 1) * 255


def clipped_zoom(img, zoom_factor):
    from scipy.ndimage import zoom as scizoom

    h = img.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top : trim_top + h, trim_top : trim_top + h]


def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03), np.arange(1, 1.36, 0.01),
        np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02), np.arange(1, 1.51, 0.02),
        np.arange(1, 1.56, 0.03),
    ][severity - 1]
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    return np.clip((x + out) / (len(c) + 1), 0, 1) * 255


def plasma_fractal(rng, mapsize=256, wibbledecay=3):
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        size = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : size : stepsize, stepsize // 2 : size : stepsize]
        ulgrid = maparray[0:size:stepsize, 0:size:stepsize]
        ltsum = drgrid + np.roll(drgrid, 1, axis=0) + ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:size:stepsize, stepsize // 2 : size : stepsize] = wibbledmean(ltsum)
        ttsum = drgrid + np.roll(drgrid, 1, axis=1) + ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2 : size : stepsize, 0:size:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x, rng, severity=1):
    c = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5),
         (4.5, 1.4), (5.0, 1.3)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    h, w = x.shape[0], x.shape[1]
    x += c[0] * plasma_fractal(rng, wibbledecay=c[1])[:h, :w][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def glass_blur(x, rng, severity=1):
    from skimage.filters import gaussian

    c = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2),
         (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)
    h, w = x.shape[0], x.shape[1]
    for _ in range(c[2]):
        for i in range(h - c[1], c[1], -1):
            for j in range(w - c[1], c[1], -1):
                dx, dy = rng.integers(-c[1], c[1], size=(2,))
                ip, jp = i + dy, j + dx
                x[i, j], x[ip, jp] = x[ip, jp], x[i, j]
    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255


def sensor_noise(image: np.ndarray, noise: int, seed: int) -> np.ndarray:
    """LIBERO-Plus `step()`: corrupt obs["agentview_image"] (robosuite convention), ids 11-50."""
    rng = np.random.default_rng(seed)
    if noise <= 10:
        raise ValueError("motion blur (ids 1-10) needs ImageMagick; not used")
    if noise <= 20:
        out = gaussian_blur(image, noise - 10)
    elif noise <= 30:
        out = zoom_blur(image, noise - 20)
    elif noise <= 40:
        out = fog(image, rng, noise - 30)
    elif noise <= 50:
        out = glass_blur(image, rng, noise - 40)
    else:
        raise ValueError(noise)
    # LIBERO-Plus feeds the float result back into the uint8 pipeline.
    return np.clip(np.asarray(out), 0, 255).astype(np.uint8)


class Renderer:
    """One MuJoCo model (one demo XML, optionally perturbed); renders recorded states."""

    def __init__(self, xml: str, size: int = 256):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0  # robosuite hides collision geoms

    def render(self, sim_state: np.ndarray) -> dict[str, np.ndarray]:
        m, d = self.model, self.data
        if len(sim_state) != 1 + m.nq + m.nv:
            raise ValueError("simulator state does not match this model")
        d.time = sim_state[0]
        d.qpos[:] = sim_state[1 : 1 + m.nq]
        d.qvel[:] = sim_state[1 + m.nq :]
        mujoco.mj_forward(m, d)
        out = {}
        for camera in CAMERAS:
            self.renderer.update_scene(d, camera, scene_option=self.option)
            out[camera] = np.ascontiguousarray(self.renderer.render()[::-1])  # robosuite convention
        return out

    def close(self):
        self.renderer.close()
