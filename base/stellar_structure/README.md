# Stellar Structure Directory

### Directory of Stellar Structure Modules

Each module in this directory defines a different stellar structure class object, each with a
modification to the stellar structure equations. Below is a list of the available stellar
structure modules and an introduction to their stellar modifications. Tne class object in each
stellar structure module is  called `StellarStructure`.

- `base_stellar_structure.py`
    - A base stellar structure equations class with no modifications.
- `grav_stellar_structure.py`
    - Stellar structure equations with a modified gravitation equation.
    - The modification includes constraining a variable, lambda, when introducing
      introducing both a 1/r term and a 1/r^3 term to the original g = GM / r^2 gravity equation.