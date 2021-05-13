"""
File to contain the necessary units.

:title: units.py

:author: Mitchell Shahen

:history: 01/05/2021
"""

# set the base SI units
kg = 1.0 # kilograms
s = 1.0 # seconds
m = 1.0 # metres
K = 1.0 # Kelvins
mol = 1.0 # moles
A = 1.0 # amperes
cd = 1.0 # candela
rad = 1.0 # radian

# additional useful derived units
J = kg * (m ** 2) / (s ** 2) # joule
C = A * s # coulomb
Hz = 1.0 / s # Hertz
N = kg * m / (s ** 2) # newton
Pa = kg / (m * (s ** 2)) # Pascal
W = kg * (m ** 2) / (s ** 3) # watt
V = kg * (m ** 2) / ((s ** 3) * A) # volt
F = ((s ** 4) * (A ** 2)) / (kg * (m ** 2)) # farad
Omega = kg * (m ** 2) / ((s ** 3) * (A ** 2)) # ohm
Wb = kg * (m ** 2) / ((s ** 2) * A) # weber
T = kg / ((s ** 2) * A) # tesla

# additional useful non-SI or non-standard units
g = 1000 * kg # grams
cm = 100 * m # centimetres
L = 0.001 * (m ** 3) # litre
AU = 149597870700 * m # astronomical unit
ly = 9.461e15 * m # lightyear
