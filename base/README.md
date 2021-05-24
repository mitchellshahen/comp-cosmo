# Base Directory of Algorithms

### Contents:
- `constants.py`
  - Includes useful and relevant constants used in cosmological calculations.
  - The module includes full compatibility with all the SI and non-SI units from `units.py`.
    
- `plot.py`
  - Includes functions for plotting graphs of useful and noteworthy stellar properties such as temperature, mass,
    density, luminosity, and pressure.
  - Available currently are plots of stellar structure, Hertzsprung-Russell disgrams, and pressure contribution plots.
    
- `solve_stellar.py`
  - Includes functions used to solve the stellar structure equations from a `stellar_structure` class like
    `StellarStructure` in `stellar_structure.py`.
  - The `solve_structure` function combines the utility from all the other functions in `solve_stellar.py`, therefore,
    this is intended to be the only function needed beyond this module.
    
- `stellar_structure.py`
  - A module containing the `StellarStructure` class designed to contain all the stellar structure equations needed to
    define a stellar body.
  - Such stellar structure equations include hydrostatic equilibrium, the energy equation, and mass continuity, among
    others.

- `store.py`
  - A module to define functions pertaining to the available data store (the `data` directory parallel to the `base`
    directory).
  - Such functions include saving data, acquiring data, and obtaining useful information about the stored data.
    
- `units.py`
  -  A module to contain variables representing units that characterize cosmological properties.
  - For example, kilograms and Kelvin are defined as are derived units such as joules and Pascals. Also, non-SI units
    such as litres and astronomical units are available.

- `util.py`
  - A module containing several functions used throughout modules in the `base` directory.
  - A few examples of functions in `util.py` include normalizing, interpolating, and extrapolating data.
