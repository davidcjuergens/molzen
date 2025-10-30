"""Periodic table of elements"""

# by period
p1 = "H He"
p2 = "Li Be B C N O F Ne"
p3 = "Na Mg Al Si P S Cl Ar"
p4 = "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
p5 = "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"

p61 = "Cs Ba"
lanthanoids = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu"
p62 = "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
p6 = f"{p61} {lanthanoids} {p62}"

p71 = "Fr Ra"
actinoids = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
p72 = "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
p7 = f"{p71} {actinoids} {p72}"

ALL_SYMBOLS = f"{p1} {p2} {p3} {p4} {p5} {p6} {p7}"
