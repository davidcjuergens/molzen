# fmt: off
# Conversion of atomic units to SI units (Szabo and Ostlund, pg. 42)
a_0     = 5.2918e-11        # length: Bohr radius (meters)
m_e     = 9.1095e-31        # mass: electron mass (kg)
e       = 1.6022e-19        # charge: elementary charge (C)
eps_a   = 4.359744722e-18   # energy: Hartree (J)
hbar    = 1.0546e-34        # angular momentum: reduced Planck constant (J*s), for higher accuracy use https://physics.nist.gov/cgi-bin/cuu/Value?hbar
ea_0    = e * a_0           # electric dipole moment (C*m)

bohr_radius         = a_0
electron_mass       = m_e
elementary_charge   = e
hartree             = eps_a

a_0_angstrom        = a_0 * 1e10  # length: Bohr radius (Angstrom)
angstrom2bohr       = 1 / a_0_angstrom

# energy conversions
hartree2kcalmol    = 627.5 # from: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
ev2kcalmol         = 23.06
hartree2ev          = 27.2107
hartree2mev         = hartree2ev * 1000

# time
au_time = hbar / hartree
fs_per_s = 1e+15
fs_per_au_time = fs_per_s * au_time 
au_time_per_fs = 1 / fs_per_au_time

# fmt: on
