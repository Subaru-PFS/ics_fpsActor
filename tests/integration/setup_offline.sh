# DAMD-195 stack driven with no hardware: cobraCharmer from the install checkout on
# alefur/tests_offline (FPGA and camera stubbed), everything else on the ticket branch.
conda activate /software/condaRoot/envs/rubin10-ics
umask 2
. ~/mhs/products/eups/default/bin/setups.sh
export PFS_SITE=L
setup -v -r /home/alefur/devel/ics/pfi/ics_fpsActor
setup -v -k -r /home/alefur/devel/ics/install/ics_cobraCharmer
setup -v -k -r /home/alefur/devel/drp/pfs_utils
setup -v -k -r /home/alefur/devel/drp/datamodel
setup -v -k -r /home/alefur/devel/ics/pfs_instdata
