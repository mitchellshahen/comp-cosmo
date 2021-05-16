"""
Plotting Module

:title: plot.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

from constants import L_sun
import matplotlib.pyplot as plt


# ---------- # ASTROPHYSICAL PLOTTING FUNCTIONS # ---------- #

def stellar_structure_plot(radius, density, temperature, mass, luminosity):
    """
    Function to plot normalized stellar density, temperature, mass, and luminosity against radius.

    :param radius: An array of radius values.
    :param density: An array of density values.
    :param temperature: An array of temperature values.
    :param mass: An array of mass values.
    :param luminosity: An array of luminosity values.
    """

    # plot the stellar density against stellar radius
    plt.plot(radius, density, label="Density", color="black", linestyle="solid")

    # plot the stellar temperature against stellar radius
    plt.plot(radius, temperature, label="Temperature", color="red", linestyle="dashed")

    # plot the stellar mass against stellar radius
    plt.plot(radius, mass, label="Mass", color="green", linestyle="dashed")

    # plot the stellar luminosity against stellar radius
    plt.plot(radius, luminosity, label="Luminosity", color="blue", linestyle="dotted")

    # set the title
    plt.title("Stellar Structure Plot")

    # set the legend
    plt.legend()

    # render the plot
    plt.show()


def hr_diagram(effect_temps, luminosities):
    """
    Function to plot a Hertzsprung-Russell diagram, that is a plot of effective surface temperature
    against relative luminosity. The effective surface temperature is in Kelvin and the luminosity
    is relative to the Sun's luminosity. The luminosity is also commonly given as a logarithm.

    :param effect_temps: An array of effective surface temperatures for many stars.
    :param luminosities: An array of non-normalized luminosities for many stars.
    """

    # normalize the luminosities with the Sun's luminosity
    for i, star_luminosity in enumerate(luminosities):
        norm_lumin = star_luminosity / L_sun
        luminosities[i] = norm_lumin

    # plot the effective surface temperatures against the stellar luminosities
    plt.plot(effect_temps, luminosities, color="black", linestyle="solid")

    # set the labels for the x-axis and y-axis
    plt.xlabel("Effective Surface Temperature (in Kelvin)")
    plt.ylabel("Relative Luminosity, L/L_sun")

    # invert the x-axis so it is decreasing in temperature going left to right
    plt.gca().invert_xaxis()

    # set the y-axis (luminosities) to be logarithmic
    plt.yscale("log")

    # set the title
    plt.title("Hertzsprung-Russell Diagram")

    # render the plot
    plt.show()
