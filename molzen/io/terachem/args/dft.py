"""DFT related args"""

from molzen.io.terachem.args.hf import FON_KWARGS

# example of WPBE functional with tuned omega
WPBE = dict(method="wpbe", rc_w=0.6)

# In this example, we will just use fon="yes", fon_method="constant"
hhtda_fon = FON_KWARGS.copy()
for pop in ["fon_target", "fon_temperature", "fon_anneal"]:
    hhtda_fon.pop(pop)
hhtda_fon["fon_method"] = "constant"

HH_TDA = dict(
    hhtda="yes",
    cisnumstates=5,
    hhtdasinglets=5,
    cisalgorithm="davidson",
    cisguessvecs=30,
    cismax=100,
    cismaxiter=1000,
    cisntos="yes",
    cisconvtol=1.0e-6,
    closed="REPLACEME",
    active="REPLACEME",
)
