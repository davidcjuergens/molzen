"""CAS related TeraChem args"""

from molzen.io.terachem.params.singlepoint import HF_ENERGY, FON_KWARGS

CAS_GRAD = dict(
    run="gradient",  # or optimize--both would require castarget and castergetmult
    castarget=None,  # zero indexed integer
    castargetmult=None,  # integer multiplicity
)

CAS_KWARGS = dict(
    maxit=200,  # default is 100
    casscf="yes",
    cassinglets=4,  # however many singlets you'd like to include
    casscfmacromaxiter=0,
    casscfmaxiter=100,  # number of "two-step" iterations, where we (1) orbital optimize, (2) CI optimize
    casscftrustmaxiter=10,
    casscfmicroconvthre=100.0,
    casscfmacroconvthre=100.0,
    casscfconvthre=1e-06,
    casscfenergyconvthre=1e-06,
    cpsacasscfmaxiter=100,
    cpsacasscfconvthre=0.001,
    cascharges="yes",
    poptype="vdd",
    ci_solver="explicit",
)
# Can also add:
# casscforbnriter=500
# casscfnriter=500
# cas_ntos="yes"

CAS_FON_ENERGY = dict(
    **HF_ENERGY,
    **CAS_KWARGS,
    **FON_KWARGS,
)
