import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panorama2gaussian.refine.mesh_extract import extract_conditioning_mesh, render_mesh


depth_path = "/root/p3-API/output/erp_4k_da360_stride1/erp_4k_depth.npy"
pano_path = "/root/p3-API/test-data/erp_4k.png"
out_path = "/root/p3-API/output/mesh_render_test.png"

depth = np.load(depth_path)
pano = Image.open(pano_path).convert("RGB")
pano = np.asarray(pano).astype(np.float32) / 255.0

print("[test] depth:", depth.shape, depth.min(), depth.max())
print("[test] pano :", pano.shape, pano.min(), pano.max())

mesh = extract_conditioning_mesh(
    depth_map=depth,
    panorama=pano,
    simplify_ratio=None,
    target_rows=256,
)

print("[test] mesh:", mesh)

img = render_mesh(
    mesh=mesh,
    camera=None,
    resolution=(512, 512),
    allow_fallback=True,
)

print("[test] render:", img.shape, img.min(), img.max(), img.mean(), img.std())

Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).save(out_path)
print("[test] saved:", out_path)