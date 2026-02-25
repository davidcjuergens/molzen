"""Coupled cluster related TeraChem args"""

# CCSD--do CIS first to get initial guess, then CCSD with Cholesky
CCSD_ARGS = dict(
    cis="yes",
    cisnumstates=3,
    ccbox="yes",
    ccbox_ccsd="yes",
    ccbox_cholesky="gpu",
    ccbox_cd_thresh=1.0e-4,
    ccbox_properties="yes",
    ccbox_ntos="yes",
    ccbox_ccsd_r_convthre=1.0e-7,
)

# EOM CCSD
EOMCCSD_ARGS = dict(
    ccbox_eomccsd_states=3,
    ccbox_eomccsd_maxsubspace=32,
    ccbox_eomccsd_r_convthre=1.0e-5,
)
