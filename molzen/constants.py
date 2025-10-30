# Conversion of atomic units to SI units (Szabo and Ostlund, pg. 42)
BOHR_RADIUS_M = 5.2918e-11  # length: Bohr radius (meters)
ELECTRON_MASS_KG = 9.1095e-31  # mass: electron mass (kg)
ELEMENTARY_CHARGE = 1.6022e-19  # charge: elementary charge (C)
JOULE_PER_HARTREE = 4.359744722e-18  # energy: Hartree (J)
HBAR = 1.0546e-34  # angular momentum: reduced Planck constant (J*s), for higher accuracy use https://physics.nist.gov/cgi-bin/cuu/Value?hbar
EA_0 = ELEMENTARY_CHARGE * BOHR_RADIUS_M  # electric dipole moment (C*m)


A_0_ANGSTROM = BOHR_RADIUS_M * 1e10  # length: Bohr radius (Angstrom)
ANGSTROM2BOHR = 1 / A_0_ANGSTROM

# energy conversions
HARTREE2KCALMOL = (
    627.5  # from: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
)
EV2KCALMOL = 23.06
HARTREE2EV = 27.2107
HARTREE2MEV = HARTREE2EV * 1000

# time
AU_TIME = HBAR / JOULE_PER_HARTREE  # atomic unit of time in seconds
FS_PER_S = 1e15
FS_PER_AU_TIME = FS_PER_S * AU_TIME
AU_TIME_PER_FS = 1 / FS_PER_AU_TIME
