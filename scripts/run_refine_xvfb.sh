#!/usr/bin/env bash
set -e

cd /root/p3-API
source .venv/bin/activate

unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
unset DISPLAY

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export MESA_LOADER_DRIVER_OVERRIDE=swrast

xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python scripts/local_pano2gs.py --refine-only \
  -o /root/p3-API/output/erp_4k_da360_stride1 \
  --ply /root/p3-API/output/erp_4k_da360_stride1/erp_4k.ply \
  --panorama /root/p3-API/test-data/erp_4k.png \
  --depth-npy /root/p3-API/output/erp_4k_da360_stride1/erp_4k_depth.npy \
  --diagnostics-dir /root/p3-API/output/erp_4k_da360_stride1/refine_diagnostics \
  --finetune-steps 0
