"""
Module to perform all the necessary functions and calculations to generate the datasets that
describe a star. Additionally, plots of the stellar structure variables are rendered.
"""

import numpy
from plot import Plotting
from solve_stellar import solve_structure
from stellar_structure import rho_index, T_index, M_index, L_index, tau_index
from store import Store
import units


def plot_stellar(all_data=None):
    """
    Function to use the Plotting class to plot all the stellar variables against
    :param all_data: A 6-Dimensional numpy ndarray containing stellar data for radius, density,
        temperature, mass, luminosity, and optical depth.
    """

    # plot the density as a function of radius
    Plotting(plot_type="line").plot(
        in_data=numpy.array([all_data[0], all_data[rho_index + 1]]),
        params={
            "title": "Plot of Stellar Density against Stellar Radius",
            "xlabel": "Stellar Radius (in m)",
            "ylabel": "Stellar Density (in kg/m**3)"
        }
    )

    # plot the temperature as a function of radius
    Plotting(plot_type="line").plot(
        in_data=numpy.array([all_data[0], all_data[T_index + 1]]),
        params={
            "title": "Plot of Stellar Temperature against Stellar Radius",
            "xlabel": "Stellar Radius (in m)",
            "ylabel": "Stellar Temperature (in K)"
        }
    )

    # plot the mass as a function of radius
    Plotting(plot_type="line").plot(
        in_data=numpy.array([all_data[0], all_data[M_index + 1]]),
        params={
            "title": "Plot of Stellar Mass against Stellar Radius",
            "xlabel": "Stellar Radius (in m)",
            "ylabel": "Stellar Mass (in kg)"
        }
    )

    # plot the luminosity as a function of radius
    Plotting(plot_type="line").plot(
        in_data=numpy.array([all_data[0], all_data[L_index + 1]]),
        params={
            "title": "Plot of Stellar Luminosity against Stellar Radius",
            "xlabel": "Stellar Radius (in m)",
            "ylabel": "Stellar Luminosity (in W)"
        }
    )

    # plot the optical density as a function of radius
    Plotting(plot_type="line").plot(
        in_data=numpy.array([all_data[0], all_data[tau_index + 1]]),
        params={
            "title": "Plot of Optical Depth against Stellar Radius",
            "xlabel": "Stellar Radius (in m)",
            "ylabel": "Optical Depth (in 1/m)"
        }
    )


def main():
    """
    Function to perform all the necessary functions to solve and plot the stellar structure
    equations.
    """

    # set the central temperature
    # T_0 = 2e7 * units.K
    T_0 = 8.23e6 * units.K

    # set the initial central density guess
    # rho_0_guess = 8.0063e4 * units.kg / (units.m ** 3)
    rho_0_guess = 5.856e4 * units.kg / (units.m ** 3)

    # set the confidence
    confidence = 0.5

    # solve the stellar structure equations and acquire the necessary data
    error, radius_arr, state_matrix = solve_structure(
        T_0,
        rho_0_guess=rho_0_guess,
        confidence=confidence,
        normalize=True
    )

    # reformat the radius to a 1-Dimensional array
    radius_arr = numpy.reshape(radius_arr, (1, -1))

    # amalgamate the radius and state datasets into a single dataset for storage
    full_data = numpy.append(radius_arr, state_matrix, axis=0)

    # save the outputted datasets for radius and state
    Store().save(data=full_data, data_filename="first_stellar_data.pickle")

    # plot all the necessary graphs describing the above generated star
    plot_stellar(all_data=full_data)


if __name__ == "__main__":
    main()
