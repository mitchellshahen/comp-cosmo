"""
Plotting module containing functions to plot various graphs in astrophysics.

:title: plot.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

import matplotlib.pyplot as plt
import numpy
import sys
sys.path.append("../") # be able to access the base directory

from base.constants import L_sun, M_sun, r_sun, rho_0_sun, T_0_sun, T_sun
from base.util import normalize_data


# ---------- # ANCILLARY PLOTTING FUNCTIONS # ---------- #

def _shade_convective(x_data=None, is_convective_arr=None):
    # add shading to the convective regions
    for i, x in enumerate(x_data):
        if is_convective_arr[i]:
            # exclude the last value to avoid an IndexError
            if i != len(x_data) - 1:
                plt.axvspan(x, x_data[i + 1], facecolor="gray", alpha=0.2)


# ---------- # ASTROPHYSICAL PLOTTING FUNCTIONS # ---------- #

def hr_diagram(effect_temps, luminosities):
    """
    Function to plot a Hertzsprung-Russell diagram, that is a plot of effective surface temperature
    against relative luminosity. The effective surface temperature is in Kelvin and the luminosity
    is relative to the Sun's luminosity. The luminosity is also commonly given as a logarithm.

    :param effect_temps: An array of effective surface temperatures for many stars.
    :param luminosities: An array of non-normalized luminosities for many stars.
    """

    # normalize the luminosities with the Sun's luminosity
    norm_luminosities = normalize_data(
        in_data=numpy.array([luminosities]),
        norm_values=[L_sun]
    )

    # set the plot size
    plt.figure(figsize=(10, 8))

    # plot the effective surface temperatures against the stellar luminosities
    plt.plot(effect_temps, norm_luminosities, color="black", linestyle="solid")

    # set the labels for the x-axis and y-axis
    plt.xlabel("Effective Surface Temperature (in Kelvin)")
    plt.ylabel(r'Relative Luminosity ($L / L_{\circ}$)')

    # invert the x-axis so it is decreasing in temperature going left to right
    plt.gca().invert_xaxis()

    # set the x-axis (effective surface temperatures) to be logarithmic
    plt.xscale("log")

    # set the y-axis (relative luminosities) to be logarithmic
    plt.yscale("log")

    # set the title
    plt.title("Hertzsprung-Russell Diagram")

    # render the plot
    plt.show()


def pressure_contributions_plot(stellar_structure, radius, density, temperature):
    """
    Function to plot the total pressure and all pressure contributions.

    :param stellar_structure: A (called) StellarStructure class containing the
        stellar structure equations.
    :param radius: An array of radius values.
    :param density: An array of density values.
    :param temperature: An array of temperature values.
    """

    # calculate the degeneracy pressure
    deg_pressure = stellar_structure.degeneracy_pressure(density)

    # calculate the gas pressure
    gas_pressure = stellar_structure.gas_pressure(
        density,
        temperature,
        mu=stellar_structure.mean_molec_weight()
    )

    # calculate the photon gas pressure
    photon_pressure = stellar_structure.photon_pressure(temperature)

    # calculate the total pressure
    total_pressure = stellar_structure.total_pressure(density, temperature)

    # calculate the surface radius to be used in normalizing the data
    surf_radius = radius[-1]

    # calculate the star's central pressure to be used in normalizing the data
    central_pressure = total_pressure[0]

    # normalize the radius array
    norm_radius = normalize_data(
        in_data=numpy.array([radius]),
        norm_values=[surf_radius]
    )

    # normalize each pressure component array
    norm_pressures = normalize_data(
        in_data=numpy.array([deg_pressure, gas_pressure, photon_pressure, total_pressure]),
        norm_values=[central_pressure, central_pressure, central_pressure, central_pressure]
    )

    # extract the normalized pressures
    norm_deg_pressure = norm_pressures[0]
    norm_gas_pressure = norm_pressures[1]
    norm_photon_pressure = norm_pressures[2]
    norm_total_pressure = norm_pressures[3]

    # set the plot size
    plt.figure(figsize=(10, 8))

    # plot each of the calculated and normalized pressures
    plt.plot(
        norm_radius,
        norm_deg_pressure,
        label="Degeneracy Pressure",
        color="blue",
        linestyle="dashed"
    )
    plt.plot(
        norm_radius,
        norm_gas_pressure,
        label="Gas Pressure",
        color="green",
        linestyle="dashed"
    )
    plt.plot(
        norm_radius,
        norm_photon_pressure,
        label="Photon Pressure",
        color="red",
        linestyle="dashed"
    )
    plt.plot(
        norm_radius,
        norm_total_pressure,
        label="Total Pressure",
        color="black",
        linestyle="solid"
    )

    # add annotations including useful stellar properties (adds context to the normalized data)
    degeneracy_pressure_text = r'$P_{deg, c} = $' + "{} Pa".format(
        format(deg_pressure[0], ".3E")
    )
    gas_pressure_text = r'$P_{gas, c} = $' + "{} Pa".format(
        format(gas_pressure[0], ".3E")
    )
    photon_pressure_text = r'$P_{phot, c} = $' + "{} Pa".format(
        format(photon_pressure[0], ".3E")
    )
    total_pressure_text = r'$P_{tot, c} = $' + "{} Pa".format(
        format(total_pressure[0], ".3E")
    )
    plt.annotate(
        "{}\n{}\n{}\n{}".format(
            degeneracy_pressure_text,
            gas_pressure_text,
            photon_pressure_text,
            total_pressure_text
        ),
        xy=(0.75, 0.4),
        xytext=(0.75, 0.4),
        textcoords="axes fraction")

    # set the title
    plt.title("Pressure Contributions Plot")

    # set the xlabel
    plt.xlabel(r'Relative Radius ($r / R_{\star}$)')

    # set the ylabel
    plt.ylabel(r'Relative Pressure ($P / P_c$)')

    # set the legend
    plt.legend(loc="upper right")

    # render the plot
    plt.show()


def stellar_structure_plot(stellar_structure, radius, density, temperature, mass, luminosity):
    """
    Function to plot normalized stellar density, temperature, mass, and luminosity against radius.

    :param stellar_structure: A (called) StellarStructure class containing the
        stellar structure equations.
    :param radius: An array of radius values.
    :param density: An array of density values.
    :param temperature: An array of temperature values.
    :param mass: An array of mass values.
    :param luminosity: An array of luminosity values.
    """

    # ---------- # NORMALIZING THE INPUT DATASETS # ---------- #

    # extract the stellar properties used to normalize the inputted datasets
    surf_radius = radius[-1]
    central_density = density[0]
    central_temp = temperature[0]
    total_mass = mass[-1]
    total_luminosity = luminosity[-1]

    # also get the star's surface temperature
    surf_temp = temperature[-1]

    # normalize the radius data
    norm_radius = normalize_data(
        in_data=numpy.array([radius]),
        norm_values=[surf_radius]
    )

    # normalize the stellar properties data
    norm_state = normalize_data(
        in_data=numpy.array([density, temperature, mass, luminosity]),
        norm_values=[central_density, central_temp, total_mass, total_luminosity]
    )

    # extract the necessary datasets from the normalized state variable
    norm_density = norm_state[0]
    norm_temperature = norm_state[1]
    norm_mass = norm_state[2]
    norm_luminosity = norm_state[3]

    # ---------- # DETERMINING THE CONVECTIVE REGIONS # ---------- #

    # iterate through the stellar properties obtaining the convective
    # and radiative temperatures at each radius value
    is_convective_list = []
    for i, radius_value in enumerate(radius):
        is_conv_value = stellar_structure.is_convective(
            radius_value,
            density[i],
            temperature[i],
            mass[i],
            luminosity[i]
        )

        is_convective_list.append(is_conv_value)

    # convert the lists of temperature values to arrays
    is_convective = numpy.array(is_convective_list)

    # ---------- # PLOTTING FUNCTIONS # ---------- #

    # set the plot size
    plt.figure(figsize=(10, 8))

    # plot the stellar density against stellar radius
    plt.plot(norm_radius, norm_density, label="Density", color="black", linestyle="solid")

    # plot the stellar temperature against stellar radius
    plt.plot(norm_radius, norm_temperature, label="Temperature", color="red", linestyle="dashed")

    # plot the stellar mass against stellar radius
    plt.plot(norm_radius, norm_mass, label="Mass", color="green", linestyle="dashed")

    # plot the stellar luminosity against stellar radius
    plt.plot(norm_radius, norm_luminosity, label="Luminosity", color="blue", linestyle="dotted")

    # add annotations including useful stellar properties to add context
    surface_radius_text = r'$R_{\star, surf} = $' + "{} m = {} ".format(
        format(surf_radius, ".3E"),
        round(surf_radius / r_sun, 3)
    ) + r'$R_{\odot, surf}$'
    central_density_text = r'$\rho_{\star, c} = $' + "{} $kg/m^3$ = {} ".format(
        format(central_density, ".3E"),
        round(central_density / rho_0_sun, 3)
    ) + r'$\rho_{\odot, c}$'
    central_temperature_text = r'$T_{\star, c} = $' + "{} K = {} ".format(
        format(central_temp, ".3E"),
        round(central_temp / T_0_sun, 3)
    ) + r'$T_{\odot, c}$'
    surface_temperature_text = r'$T_{\star, surf} = $' + "{} K = {} ".format(
        format(surf_temp, ".3E"),
        round(surf_temp / T_sun, 3)
    ) + r'$T_{\odot, surf}$'
    total_mass_text = r'$M_{\star} = $' + "{} kg = {} ".format(
        format(total_mass, ".3E"),
        round(total_mass / M_sun, 3)
    ) + r'$M_{\odot}$'
    total_luminosity_text = r'$L_{\star} = $' + "{} W = {} ".format(
        format(total_luminosity, ".3E"),
        round(total_luminosity / L_sun, 3)
    ) + r'$L_{\odot}$'
    plt.annotate(
        "{}\n{}\n{}\n{}\n{}\n{}".format(
            surface_radius_text,
            central_density_text,
            central_temperature_text,
            surface_temperature_text,
            total_mass_text,
            total_luminosity_text
        ),
        xy=(0.6, 0.4),
        xytext=(0.6, 0.4),
        textcoords="axes fraction")

    # shade in the areas where convective temperature forces dominate
    _shade_convective(x_data=norm_radius, is_convective_arr=is_convective)

    # set the title
    plt.title("Stellar Structure Plot")

    # set the xlabel
    plt.xlabel(r'Relative Radius ($r / R_{\star}$)')

    # set the ylabel
    plt.ylabel(r'$\rho / \rho_c$, $T / T_c$, $M / M_{\star}$, $L / L_{\star}$')

    # set the legend
    plt.legend(loc="upper right")

    # render the plot
    plt.show()
