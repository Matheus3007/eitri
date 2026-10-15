#!/usr/bin/env bash
# Run one OpenCourant (community fork of OpenRadioss, AGPL v3) tool with its runtime environment.
#   ./tools/opencourant.sh starter|engine|anim_to_vtk|th_to_csv [args...]
# Install: the linux64 zip from https://opencourant.org/downloads/ extracted to $OPENCOURANT_PATH
# (default ~/opt/OpenCourant). Run from the directory holding the deck (the solver writes next to it).
set -euo pipefail
OC="${OPENCOURANT_PATH:-$HOME/opt/OpenCourant}"
case "${1:-}" in
  starter) exe=starter_linux64_gf ;;
  engine) exe=engine_linux64_gf ;;
  anim_to_vtk) exe=anim_to_vtk_linux64_gf ;;
  th_to_csv) exe=th_to_csv_linux64_gf ;;
  *) echo "usage: $0 starter|engine|anim_to_vtk|th_to_csv [args...]" >&2; exit 2 ;;
esac
shift
[ -x "$OC/exec/$exe" ] || chmod +x "$OC/exec/"*  # extracted with python's zipfile: no exec bits
export OPENCOURANT_PATH="$OC"
export RAD_CFG_PATH="$OC/hm_cfg_files"
export LD_LIBRARY_PATH="$OC/extlib/hm_reader/linux64:$OC/extlib/h3d/lib/linux64:${LD_LIBRARY_PATH:-}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-400m}"
exec "$OC/exec/$exe" "$@"
