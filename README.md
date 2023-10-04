# Computational Cosmology

## Library and Environment for Solving and Modelling the Stellar Structure Equations

### Introduction

Available are modules for calculating and modelling complex cosmological processes including solving the
structure of stars garnering stellar properties such as density, temperature, mass, and luminosity.

### Motivation

I endeavoured to develop this project as a recent graduate in Honours Physics from the University of Waterloo to further my knowledge of stellar formation and programming in scientific environments. This project is an extension and adaptation of completed coursework from my time at the University of Waterloo that will go beyond the scope of the course from which this project was derived.


### Acknowledgements and Collaborations

The foundation of this project was developed as an end-of-term project as part of the PHYS 375 coursework in the University of Waterloo's Physics and Astronomy department. The project was formulated and overseen by [Dr. Avery Broderick](https://perimeterinstitute.ca/people/avery-broderick), lecturing professor at UW. As a student of this course, the foundation of this project was developed in collaboration with students and colleagues Amaar Quadri, Omar Auda, Kevin Djuric, Shivani Hegde, Frederick Hobel, Jedri de Luna, and Meagan Stewart. The completed code, project outline, and final results completed as part of the original coursework is included on Amaar's [github](https://github.com/amaarquadri/StarsModifiedGravity). Amaar was also instrumental in developing the main code base for the PHYS 375 project.

### Directories

- `base`
    - The primary directory of stellar solution and plotting modules
- `data`
    - The directory used to contain stellar solution data, by default.
- `test`
    - The directory containing testing modules to verify and validate modules in the `base` directory.

### Set Up and Installation

Included in the `comp-cosmo` directory is the `setup.py` file used to create a python virtual environment containing
all the necessary modules and dependencies from matplotlib, numpy, and scipy.
Note: When running this environment on Linux, ensure that the `python3-tk` package is installed by running the following command.

```sh
sudo apt-get install python3-tk
```

### Available Functionality

Two types of stellar solutions are available, single star generation and stellar sequence generation. Executing
the `main.py` module, by running the following command while in the `comp-cosmo` directory, will give the option
of selecting a stellar solution to perform and a list of the modifications that may be applied.

(Windows)

```sh
python main.py
```

(Linux)

```sh
python3 main.py
```

The single star generation method involves solving the stellar structure equations, included below, using a
fixed central (core) temperature, performing the integration solution, and checking the results against an
intended optical density metric. Included below is a more detailed description of this process.

The stellar sequence generation method involves solving numerous stars at various central temperatures, as
is done for a single star generation, and maintaining useful (global) information pertaining to each star.
Included below is a more detailed description of this process.

### Stellar Structure Equations

- Hydrostatic Equilibrium
    - dP/dr = - G * mass * density / radius^2

- Energy Equation
    - dL/dr = 4 * pi * radius^2 * density * energy generation rate

- Mass Equation
    - dM/dr = 4 * pi * radius^2 * density

- Temperature Gradient
    - dT/dr = - min(convective temperature, radiative temperature)

- Optical Density Gradient
    - dtau/dr = mean opacity * density

### Stellar Structure Modifications

The stellar structure equations are included in a class object always called `StellarStructure`. However,
additional stellar structure classes are available to investigate the impacts of varying the stellar structure
equations. Eahc stellar structure class is included in its own module while the name of the module reflects
the structure modification, if any. All stellar structure modules are stored in the `base\stellar_structure`
directory. The unmodified stellar structure class is included in `base_stellar_structure.py`. Additional
stellar structure modules and their included modifications are included below.

- `grav_stellar_structure.py`
    - Gravitational contributions are modified to include a 1/r^3 term (scaled by lambda), and a 1/r term (scaled by
      Lambda) in addition to the original 1/r^2 term.
    - g = G * M / r^2 --> g = G * M * (1 + lambda / r) * (1 + r / Lambda) / r^2

### Single Star Generation

The process for solving the stellar structure equations for a single star is outlined below:

1. Select a central (core) temperature.
2. Select a central (core) density estimate.
3. Set initial (core/central) properties for the remaining stellar variables (luminosity, mass, optical density)
   based on the central (core) temperature and density.
4. Solve the stellar variables using numerical integration until the optical density equates to 2/3 (a suitable
   approximation for identifying the star's surface).
5. Look at the calculated central (core) density and compare it to the initial estimate.
6. If the difference between the estimated and calculated densities exceed the allowed error, amend the
   initial estimate and repeat steps 3-6.
7. If the difference between the estimated and calculated densities are within the allowed error, output the
   calculated stellar variables.

### Stellar Sequence Generation

1. Set the initial and final central (core) temperatures to survey.
2. Set the central (core) density estimate for the first star in the sequence.
3. Solve the first star in the sequence as was done for the single star generation.
4. Use the previous central (core) density estimate(s) to extrapolate the density estimate for the next star.
5. Use the next central (core) temperature in the survey and solve the next star in the sequence as was
   done for the single star generation.
6. Collect useful information about the solved star (surface radius, total mass, total luminosity,
   surface temperature, etc.)
7. Repeat steps 4-6 for all remaining stars in the sequence.

### Main Executable Module

`main.py` is the primary executable module for this environment. Doing so will execute a script that prompts
the user to select the type of solution to generate, single star or stellar sequence are currently the only
supported calculation types. After selecting the solution method, the necessary parameters pertaining to the
selected method must be specified or allowed to revert to the default values for each parameter. Following
this, the solution calculation will commence until a satisfactory solutions is achieved. The resulting
solution data is then saved (unless otherwise specified) and plots describing the solution is rendered.

### Common Errors and Troubleshooting

- Erroneous Central Density Estimates
    - Description: The single star solution procedure requires a central density estimate be provided and
      sequentially amended until a central density estimate achieves a certain level of precision. If the
      original central density estimate is too far from the actual, previously unknown, value, the solution
      generated by this estimate will be even further from the actual solution. Therefore, the amended value
      will be even more askew than the original estimate causing an overall divergence from the intended
      solution.
    - Solution: For a single star solution, one of the input parameters specified by the user is the
      original central density estimate. Therefore, re-running the program multiple times with several
      varying central density estimates will produce many results of varying plausibilities. For a stellar
      sequence solution, aside from solving the first star, central density estimates are extrapolated based
      on data from previously solved stars. Omitting the possibility of a bug in the extrapolation
      calculations, erroneous results are likely due to skewed results in solving the first star using the
      user-defined central density estimate. The solution is therefore identical to that for a single star
      solution.
