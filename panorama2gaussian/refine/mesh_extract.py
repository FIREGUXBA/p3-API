"""从深度图提取网格，供 GSFixer 双条件使用。"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_conditioning_mesh(
    depth_map,
    panorama,
    simplify_ratio=None,
    target_rows=384,
    max_depth=500,
    min_depth=0.01,
    depth_edge_ratio=1.35,
):
    """从等距柱状深度直接规则三角化得到带纹理网格（无 Poisson）。

    对规则 ERP 深度栅格：相邻 2×2 像素块分成两个三角形，深度比超过
    ``depth_edge_ratio`` 时不连边以减轻跨遮挡拖影。比 Poisson 快得多，
    且更接近可见表面，适合 conditioning 几何。
    """
    import trimesh

    depth_map = np.asarray(depth_map, dtype=np.float64)
    pan = np.asarray(panorama)
    if pan.ndim == 2:
        pan = np.stack([pan, pan, pan], axis=-1)
    elif pan.shape[-1] == 4:
        pan = pan[..., :3]
    elif pan.shape[-1] == 1:
        pan = np.repeat(pan, 3, axis=-1)
    pan = pan[..., :3]

    if pan.shape[0] != depth_map.shape[0] or pan.shape[1] != depth_map.shape[1]:
        raise ValueError(
            f"panorama {pan.shape[:2]} 与 depth_map {depth_map.shape[:2]} 空间尺寸不一致"
        )

    h, w = depth_map.shape
    stride = max(1, h // target_rows)

    depth = depth_map[::stride, ::stride]
    pano = pan[::stride, ::stride, :3]

    hs, ws = depth.shape

    # SPAG 约定：θ 从左到右递减，φ 为余纬 0→π；与旧 Poisson 路径一致
    theta = (1.0 - np.linspace(0, 1, ws, endpoint=False)) * 2 * np.pi
    phi = np.linspace(0, np.pi, hs)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = depth * np.sin(phi_grid) * np.cos(theta_grid)
    y = depth * np.cos(phi_grid)
    z = -depth * np.sin(phi_grid) * np.sin(theta_grid)

    valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)

    points_full = np.stack([x, y, z], axis=-1)
    index_map = -np.ones((hs, ws), dtype=np.int64)
    vertices = points_full[valid]
    colors = pano[valid]

    index_map[valid] = np.arange(len(vertices), dtype=np.int64)

    faces = []

    def depth_coherent(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if np.any(vals <= min_depth) or np.any(vals >= max_depth):
            return False
        dmin = float(vals.min())
        dmax = float(vals.max())
        return dmax / max(dmin, 1e-6) < depth_edge_ratio

    for i in range(hs - 1):
        for j in range(ws):
            j_next = (j + 1) % ws

            ids = [
                index_map[i, j],
                index_map[i + 1, j],
                index_map[i, j_next],
                index_map[i + 1, j_next],
            ]

            if min(ids) < 0:
                continue

            d00 = depth[i, j]
            d10 = depth[i + 1, j]
            d01 = depth[i, j_next]
            d11 = depth[i + 1, j_next]

            if depth_coherent([d00, d10, d01]):
                faces.append([ids[0], ids[1], ids[2]])

            if depth_coherent([d10, d11, d01]):
                faces.append([ids[1], ids[3], ids[2]])

    if len(vertices) < 100 or len(faces) < 100:
        logger.warning("有效顶点/面过少，无法提取网格")
        return None

    vertex_colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=vertex_colors,
        process=False,
    )

    if simplify_ratio is not None and 0 < simplify_ratio < 1:
        target_faces = max(100, int(len(mesh.faces) * simplify_ratio))
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as e:
            logger.warning(f"Trimesh 网格简化失败：{e}")

    logger.info(
        f"已提取网格（ERP 规则三角化）：{len(mesh.vertices)} 顶点，{len(mesh.faces)} 面"
    )
    return mesh


def render_mesh(
    mesh,
    camera,
    resolution=(512, 512),
    point_radius=None,
    fill_holes=True,
    allow_fallback=False,
):
    """用纯 CPU 将 mesh 顶点投影到当前 CameraPose，生成 GSFixer mesh condition。

    注意：
    - 不依赖 OpenGL / GLU / pyglet / Xvfb；
    - 会真正使用 camera；
    - resolution 按当前项目习惯解释为 (H, W)；
    - 输出 (H, W, 3) float32, [0, 1]。
    """
    if mesh is None:
        msg = "render_mesh: mesh is None"
        if allow_fallback:
            logger.warning(msg + "，返回灰色占位图")
            h, w = resolution
            return np.ones((h, w, 3), dtype=np.float32) * 0.5
        raise RuntimeError(msg)

    if camera is None:
        msg = "render_mesh: camera is None，无法按修复视角渲染 mesh condition"
        if allow_fallback:
            logger.warning(msg + "，返回灰色占位图")
            h, w = resolution
            return np.ones((h, w, 3), dtype=np.float32) * 0.5
        raise RuntimeError(msg)

    h, w = int(resolution[0]), int(resolution[1])

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise RuntimeError("render_mesh: mesh has no vertices")

    # 读取顶点颜色
    colors = None
    try:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == vertices.shape[0]:
                colors = vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        colors = None

    if colors is None:
        colors = np.ones((vertices.shape[0], 3), dtype=np.float32) * 0.8

    colors = np.clip(colors, 0.0, 1.0)

    # CameraPose -> camera basis
    pos = np.asarray(camera.position, dtype=np.float64)
    look_at = np.asarray(camera.look_at, dtype=np.float64)
    up_world = np.asarray(camera.up, dtype=np.float64)

    forward = look_at - pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-8)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # world -> camera
    rel = vertices - pos[None, :]
    x_cam = rel @ right
    y_up = rel @ up
    z_cam = rel @ forward

    # 只保留相机前方点
    valid = z_cam > 1e-4
    if not np.any(valid):
        raise RuntimeError("render_mesh: no mesh vertices in front of camera")

    x_cam = x_cam[valid]
    y_up = y_up[valid]
    z_cam = z_cam[valid]
    colors = colors[valid]

    # 透视投影。camera.fov_deg 按垂直 FOV 处理。
    f = h / (2.0 * np.tan(np.radians(camera.fov_deg) / 2.0))
    cx = w / 2.0
    cy = h / 2.0

    u = f * (x_cam / z_cam) + cx
    v = -f * (y_up / z_cam) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not np.any(inside):
        raise RuntimeError("render_mesh: projected mesh is outside image")

    ui = ui[inside]
    vi = vi[inside]
    z_cam = z_cam[inside]
    colors = colors[inside]

    # 近处覆盖远处
    order = np.argsort(z_cam)[::-1]  # far -> near，后画的近点覆盖远点

    image = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=bool)

    if point_radius is None:
        # 512 分辨率时约 3~4 像素，适合 conditioning
        point_radius = max(1, int(round(h / 256)))

    r = int(point_radius)

    for idx in order:
        x = ui[idx]
        y = vi[idx]
        c = colors[idx]

        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        image[y0:y1, x0:x1, :] = c
        coverage[y0:y1, x0:x1] = True

    # 用最近可见颜色填补点投影缝隙，避免 mesh condition 又出现大面积黑/灰洞
    if fill_holes and np.any(coverage) and np.any(~coverage):
        try:
            from scipy.ndimage import distance_transform_edt

            empty = ~coverage
            _, inds = distance_transform_edt(empty, return_indices=True)
            filled = image[inds[0], inds[1]]
            image[empty] = filled[empty]
        except Exception as e:
            logger.warning(f"render_mesh: 最近邻填洞失败：{e}")

    std = float(image.std())
    if std < 1e-4:
        raise RuntimeError(
            f"render_mesh: mesh condition nearly constant, std={std:.8f}"
        )

    logger.debug(
        "render_mesh cpu_splat: vertices=%d projected=%d coverage=%.3f std=%.4f",
        len(vertices),
        len(ui),
        float(coverage.mean()),
        std,
    )

    return np.clip(image, 0.0, 1.0).astype(np.float32)
