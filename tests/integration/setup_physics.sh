# Full DAMD-195 stack: fpsActor, pfs_utils, pfs_instdata and datamodel on the ticket
# branch; cobraCharmer on master (DAMD-195 has no branch there).
conda activate /software/condaRoot/envs/rubin10-ics
umask 2
. ~/mhs/products/eups/default/bin/setups.sh
export PFS_SITE=L
setup -v -r /home/alefur/devel/ics/pfi/ics_fpsActor
setup -v -k -r /home/alefur/devel/ics/pfi/ics_cobraCharmer
setup -v -k -r /home/alefur/devel/drp/pfs_utils
setup -v -k -r /home/alefur/devel/drp/datamodel
setup -v -k -r /home/alefur/devel/ics/pfs_instdata
