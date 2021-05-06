"""
File to set up all necessary universal constants.

:title: constants.py

:author: Mitchell Shahen

:history: 01/05/2021
"""

# import the units to properly define the constants
import units

# gravitational constant (in metres cubed per kilogram per second squared
G = 6.6743e-11 * (units.m ** 3 / (units.kg * units.s ** 2))

# speed of light (in metres per second)
c = 299792458 * (units.m / units.s)

# planck's constants (in Joule-seconds)
h = 6.62607015e-34 * (units.J / units.s)

# planck's constants (in Joule-seconds)
h_bar = 1.054571817e-34 * (units.J / units.s)

# elementary charge (in Coulombs)
e = 1.602176634e-19 * (units.C)

# boltzmann constant (in Joules per Kelvin)
k = 1.380649e-23 * (units.J / units.K)

# Stefan-Boltzmann constant (in Watts per unit area per Kelvin to the fourth power)
sigma = 5.670374419e-8 * (units.W / (units.m ** 2 * units.K ** 4))

# radiation constant (in Joules per unit volume per Kelvin to the fourth power)
a = 7.5657e-16 * (units.J / (units.m ** 3 * units.K ** 4))

# adiabatic index for an ideal gas (unitless)
gamma = 5 / 3

# proton mass
m_p = 1.6726219e-27 * (units.kg)

# electron mass
m_e = 9.10938356e-31 * (units.kg)
