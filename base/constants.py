"""
File to set up all necessary universal constants.

:title: constants.py

:author: Mitchell Shahen

:history: 01/05/2021
"""

from base.units import m, kg, s, J, C, K, W

# ---------- # UNIVERSAL CONSTANTS # ---------- #

# gravitational constant (in metres cubed per kilogram per second squared
G = 6.6743e-11 * (m ** 3 / (kg * s ** 2))

# speed of light (in metres per second)
c = 299792458 * (m / s)

# planck's constants (in Joule-seconds)
h = 6.62607015e-34 * (J / s)

# planck's constants (in Joule-seconds)
h_bar = 1.054571817e-34 * (J / s)

# elementary charge (in Coulombs)
e = 1.602176634e-19 * (C)

# boltzmann constant (in Joules per Kelvin)
k = 1.380649e-23 * (J / K)

# Stefan-Boltzmann constant (in Watts per unit area per Kelvin to the fourth power)
sigma = 5.670374419e-8 * (W / (m ** 2 * K ** 4))

# radiation constant (in Joules per unit volume per Kelvin to the fourth power)
a = 7.5657e-16 * (J / (m ** 3 * K ** 4))

# adiabatic index for an ideal gas (unitless)
gamma = 5 / 3

# proton mass (in kilograms)
m_p = 1.6726219e-27 * (kg)

# electron mass (in kilograms)
m_e = 9.10938356e-31 * (kg)

# ---------- # SOLAR PROPERTIES # ---------- #

# solar luminosity (in Watts)
L_sun = 3.846e26 * (W)

# solar mass (in kilograms)
M_sun = 1.98847e30 * (kg)

# solar radius (in metres)
r_sun = 6.96342e8 * (m)

# solar central density (in kg per cubic metre)
rho_0_sun = 1.60e5 * (kg / (m ** 3))

# solar central temperature
T_0_sun = 1.5e7 * (K)

# solar surface temperature
T_sun = 5778 * (K)
