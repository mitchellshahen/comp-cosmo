"""
File to set up all necessary universal constants.

:title: constants.py

:author: Mitchell Shahen

:history: 01/05/2021
"""

import sys
sys.path.append("../") # be able to access the base directory

import base.units as units

# ---------- # UNIVERSAL CONSTANTS # ---------- #

# gravitational constant (in metres cubed per kilogram per second squared
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?bg`
G = 6.67430e-11 * (units.m ** 3 / (units.kg * units.s ** 2))

# speed of light (in metres per second)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?c`
c = 299792458 * (units.m / units.s)

# planck's constants (in Joule-seconds)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?h`
h = 6.62607015e-34 * (units.J / units.s)

# planck's constants (in Joule-seconds)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?hbar`
h_bar = 1.054571817e-34 * (units.J / units.s)

# elementary charge (in Coulombs)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?e`
e = 1.602176634e-19 * (units.C)

# boltzmann constant (in Joules per Kelvin)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?k`
k = 1.380649e-23 * (units.J / units.K)

# Stefan-Boltzmann constant (in Watts per unit area per Kelvin to the fourth power)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?sigma`
sigma = 5.670374419e-8 * (units.W / (units.m ** 2 * units.K ** 4))

# radiation constant (in Joules per unit volume per Kelvin to the fourth power)
# the radiation constant is also given as a = 4 * sigma / c
# reference: `https://scienceworld.wolfram.com/physics/RadiationConstant.html`
a = 7.5657e-16 * (units.J / (units.m ** 3 * units.K ** 4))

# adiabatic index for an ideal/monatomic gas (unitless)
# reference: `hyperphysics.phy-astr.gsu.edu/hbase/thermo/adiab.html`
gamma = 5 / 3

# proton mass (in kilograms)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?mp`
m_p = 1.67262192369e-27 * (units.kg)

# electron mass (in kilograms)
# reference: `https://www.physics.nist.gov/cgi-bin/cuu/Value?me`
m_e = 9.1093837015e-31 * (units.kg)

# ---------- # SOLAR PROPERTIES # ---------- #

# solar luminosity (in Watts)
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
L_sun = 3.828e26 * (units.W)

# solar mass (in kilograms)
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
M_sun = 1.988500e30 * (units.kg)

# solar radius (in metres)
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
r_sun = 6.95700e8 * (units.m)

# solar central density (in kg per cubic metre)
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
rho_0_sun = 1.622e5 * (units.kg / (units.m ** 3))

# solar central temperature
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
T_0_sun = 1.571e7 * (units.K)

# solar surface effective temperature
# reference: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html`
T_sun = 5772 * (units.K)
