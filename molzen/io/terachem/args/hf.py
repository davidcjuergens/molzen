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
# Other options: fon_method="constant"
FON_KWARGS = dict(
    fon="yes",
    fon_anneal="no",
    fon_target=0.2,
    fon_temperature=0.2,
)
