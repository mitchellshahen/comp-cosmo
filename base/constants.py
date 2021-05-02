"""
File to set up all necessary universal constants.

:title: constants.py

:author: Mitchell Shahen

:history: 01/05/2021
"""

# import the units to properly define the constants
from base import units

# speed of light (in metres per second)
c = 299792458 * (units.m / units.s)

# planck's constants (in Joule-seconds)
h = 6.62607015e-34 * (units.J / units.s)

# elementary charge (in Coulombs)
e = 1.602176634e-19 * (units.C)

# boltzmann constant
k = 1.380649e-23 * (units.J / units.K)
