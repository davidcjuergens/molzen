"""Single point job parameters"""

# Basic HF energy call
HF_ENERGY = dict(
    run="energy",
    guess="generate",
    basis="6-31gss",
    method="HF",
    threall=1e-14,
    convthre=1e-7,
    precision="mixed",
)

# Fractional occupation number (FON) specific
FON_KWARGS = dict(
    fon="yes",
    fon_anneal="no",
    fon_target=0.2,
    fon_temperature=0.2,
)

CAS_GRAD= dict(
    run="gradient", # or optimize--both would require castarget and castergetmult
    castarget=1,  # Now we are doing optimization on S1
    castargetmult=1,
)

CAS_KWARGS = dict(
    maxit=200,  # default is 100
    casscf="yes",
    cassinglets=4, # however many singlets you'd like to include
    casscfmacromaxiter=0,
    casscfmaxiter=100, # number of "two-step" iterations, where we (1) orbital optimize, (2) CI optimize
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
    cas_ntos="yes", # more expensive at the end due to SVD the transition density matrix
)
# Can also add:
# casscforbnriter=500
# casscfnriter=500

CAS_FON_ENERGY = dict(
    **HF_ENERGY,
    **CAS_KWARGS,
    **FON_KWARGS,
)
