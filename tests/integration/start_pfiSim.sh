#!/bin/bash
# The physics FPGA (localhost:4001) and its truth server (localhost:4002) for the live
# offline stack.  Start it, then fps with COBRACHARMER_FPGA_HOST=localhost and mcs with
# MCS_SIM_TRUTH=localhost:4002.  Arguments go to pfiSimServer.py (--seed, --phiGain, ...).
conda activate /software/condaRoot/envs/rubin10-ics
umask 2
. ~/mhs/products/eups/default/bin/setups.sh
export PFS_SITE=L
setup -v -r /home/alefur/devel/ics/install/ics_fpsActor
setup -v -k -r /home/alefur/devel/ics/install/ics_cobraCharmer
setup -v -k -r /home/alefur/devel/drp/pfs_utils
setup -v -k -r /home/alefur/devel/drp/datamodel
setup -v -k -r /home/alefur/devel/ics/pfs_instdata
cd "$(dirname "$0")"
exec python pfiSimServer.py "$@"
