# Conversion of atomic units to SI units (Szabo and Ostlund, pg. 42)

# Length
BOHR_RADIUS_M = 5.2918e-11  # length: Bohr radius (meters)
A_0_ANGSTROM = BOHR_RADIUS_M * 1e10  # length: Bohr radius (Angstrom)
ANGSTROM2BOHR = 1 / A_0_ANGSTROM

# Mass
ELECTRON_MASS_KG = 9.1095e-31  # mass: electron mass (kg)

# Charge
ELEMENTARY_CHARGE = 1.6022e-19  # charge: elementary charge (C)

# Electric Dipole Moment
EA_0 = ELEMENTARY_CHARGE * BOHR_RADIUS_M  # electric dipole moment (C*m)

# Angular Momentum
HBAR = 1.0546e-34  # angular momentum: reduced Planck constant (J*s), for higher accuracy use https://physics.nist.gov/cgi-bin/cuu/Value?hbar


# Energy
JOULE_PER_HARTREE = 4.359744722e-18  # energy: Hartree (J)
# from: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
HARTREE2KCALMOL = 627.5
hartree_to_kcalmol = HARTREE2KCALMOL
EV2KCALMOL = 23.06
ev_to_kcalmol = EV2KCALMOL
HARTREE2EV = 27.2107
hartree_to_ev = HARTREE2EV
HARTREE2MEV = HARTREE2EV * 1000
hartree_to_mev = HARTREE2MEV
# from NIST: https://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=hr&To=minv&Action=Convert+value+and+show+factor
HARTREE2INVERSE_M = 2.1947463136314e7  # Hartree --> inverse meter
hartree_to_inverse_m = HARTREE2INVERSE_M
HARTREE2INVERSE_CM = HARTREE2INVERSE_M / 100
hartree_to_inverse_cm = HARTREE2INVERSE_CM


# time
AU_TIME = HBAR / JOULE_PER_HARTREE  # atomic unit of time in seconds
FS_PER_S = 1e15
FS_PER_AU_TIME = FS_PER_S * AU_TIME
AU_TIME_PER_FS = 1 / FS_PER_AU_TIME
