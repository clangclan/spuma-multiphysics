#!/bin/bash
set -eo pipefail
source /home/jsw/cae-gpu-pr1-compatible/env.sh
cd "$WM_PROJECT_DIR"
(cd wmake/src && make -j8)
wmakeLnInclude -u src/OpenFOAM
wmakeLnInclude -u src/OSspecific/POSIX
WM_NCOMPPROCS=8 src/OSspecific/POSIX/Allwmake
wmake -j8 libso src/Pstream/dummy
for part in OpenFOAM fileFormats surfMesh meshTools finiteVolume mesh/blockMesh mesh/extrudeModel dynamicMesh; do
    wmake -j8 libso "src/$part"
done
wmake -j8 applications/utilities/mesh/generation/blockMesh
blockMesh -help
