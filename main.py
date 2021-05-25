"""
Module to perform all the necessary functions and calculations to generate the datasets that
describe a star. Additionally, plots of the stellar structure variables are rendered.

:title: main.py

:author: Mitchell Shahen

:history: 10/05/2021
"""

import numpy

from base.plot import hr_diagram, pressure_contributions_plot, stellar_structure_plot
from base.solve_stellar import solve_structure
from base.stellar_structure import StellarStructure, L_index, M_index, rho_index, T_index
from base.store import Store
import base.units as units
from base.util import extrapolate

# specify the set of Stellar Structure Equations to use as well as any modifications
stellar_structure = StellarStructure()


def _generate_star():
    """
    Function to perform all the necessary functions to solve and plot the stellar structure
    equations.
    """

    # set the central temperature by providing a value or using the default
    temp_input = input(
        "\nSet the Star's Central Temperature, in Kelvin, as a positive non-zero number "
        "(Press [Enter] to Use the Default, 15000000 Kelvin) >>> "
    )
    if len(temp_input) != 0:
        T_0 = (float(temp_input) if float(temp_input) > 0 else 1.5e7) * units.K
    else:
        T_0 = 1.5e7 * units.K

    # set the initial central density guess by providing a value or using the default
    rho_0_input = input(
        "\nSet the Star's Central Density, in kg/m^3, as a positive non-zero number "
        "(Press [Enter] to Use the Default, 100000 kg/m^3) >>> "
    )
    if len(rho_0_input) != 0:
        rho_0_guess = (
            float(rho_0_input) if float(rho_0_input) > 0 else 1e5
        ) * units.kg / (units.m ** 3)
    else:
        rho_0_guess = 1e5 * units.kg / (units.m ** 3)

    # set the confidence by providing a value or using the default
    conf_input = input(
        "\nSet the Confidence Level, a number between 0.5 and 1.0, to Use When Solving the "
        "Stellar Structure Equations (Press [Enter] to Use the Default, 0.5) >>> "
    )
    if len(conf_input) != 0:
        confidence = float(conf_input) if 0.5 <= float(conf_input) < 1.0 else 0.5
    else:
        confidence = 0.5

    # set the decision of whether or not to save the data by providing a value or using the default
    save_input = input(
        "\nSet Decision of if Data is Saved, [0] for False or [1] for True "
        "(Press [Enter] to Use the Default, False) >>> "
    )
    if len(save_input) != 0:
        save_data = bool(int(save_input)) if int(save_input) in [0, 1] else False
    else:
        save_data = False

    # solve the stellar structure equations and acquire the necessary data
    lumin_error, radius_arr, state_matrix = solve_structure(
        stellar_structure,
        T_0=T_0,
        rho_0_guess=rho_0_guess,
        confidence=confidence
    )

    # reformat the radius and add it to the state to form a single dataset for storage
    full_data = numpy.append(
        numpy.reshape(radius_arr, (1, -1)),
        state_matrix,
        axis=0
    )

    # save the outputted datasets for radius and state
    if save_data:
        Store().save(data=full_data, data_filename="stellar_data.pickle")

    # plot all the necessary graphs describing the above generated star
    stellar_structure_plot(
        radius=full_data[0],
        density=full_data[rho_index + 1],
        temperature=full_data[T_index + 1],
        mass=full_data[M_index + 1],
        luminosity=full_data[L_index + 1]
    )

    # plot the pressure and pressure contributions
    pressure_contributions_plot(
        stellar_structure=stellar_structure,
        radius=full_data[0],
        density=full_data[rho_index + 1],
        temperature=full_data[T_index + 1]
    )


def _generate_star_sequence():
    """
    Function to generate a star sequence: generating several stars and saving only each star's most
    important properties.
    """

    # set the number of stars to generate
    N_input = input(
        "\nSet the Number of Stars to Generate (Press [Enter] to Use the Default, 50) >>> "
    )
    if len(N_input) != 0:
        N = int(N_input) if int(N_input) > 0 else 50
    else:
        N = 50

    # set the initial central temperature to survey
    T_0_i_input = input(
        "\nSet the Central Temperature, in Kelvin, of the First Star in the Sequence "
        "(Press [Enter] to Use the Default, 300000 Kelvin) >>> "
    )
    if len(T_0_i_input) != 0:
        T_0_i = (float(T_0_i_input) if float(T_0_i_input) > 0 else 3e5) * units.K
    else:
        T_0_i = 3e5 * units.K

    # set the final central temperature to survey
    T_0_f_input = input(
        "\nSet the Central Temperature, in Kelvin, of the First Star in the Sequence "
        "(Press [Enter] to Use the Default, 30000000 Kelvin) >>> "
    )
    if len(T_0_f_input) != 0:
        T_0_f = (float(T_0_f_input) if float(T_0_f_input) > 0 else 3e7) * units.K
    else:
        T_0_f = 3e7 * units.K

    # set the confidence by providing a value or using the default
    conf_input = input(
        "\nSet the Confidence Level, a number between 0.5 and 1.0, to Use When Solving the "
        "Stellar Structure Equations (Press [Enter] to Use the Default, 0.5) >>> "
    )
    if len(conf_input) != 0:
        confidence = float(conf_input) if 0.5 <= float(conf_input) < 1.0 else 0.5
    else:
        confidence = 0.5

    # set the decision of whether or not to save the data by providing a value or using the default
    save_input = input(
        "\nSet Decision of if Data is Saved, [0] for False or [1] for True "
        "(Press [Enter] to Use the Default, False) >>> "
    )
    if len(save_input) != 0:
        save_data = bool(int(save_input)) if int(save_input) in [0, 1] else False
    else:
        save_data = False

    # generate an array of temperature values to generate stars with
    all_cen_temp = numpy.linspace(T_0_i, T_0_f, N)

    # run the solve_structure function for the first star using the default central density guess;
    # this is done to create non-empty arrays of central temperatures and central densities that can
    # be used to predict viable central densities for all the proceeding central temperature values
    print("\nSolving Star 1")
    __, first_radius_arr, first_state_matrix = solve_structure(
        stellar_structure,
        T_0=T_0_i,
        rho_0_guess=1e5 * units.kg / (units.m ** 3),
        confidence=confidence
    )

    # create an array to contain all the important properties from each star;
    # include the important properties from the first generated star
    important_properties = numpy.array([
        [first_radius_arr[-1]], # surface radii
        [first_state_matrix[rho_index][0]], # central densities
        [first_state_matrix[T_index][0]], # central temperature
        [first_state_matrix[T_index][-1]], # surface temperatures
        [first_state_matrix[M_index][-1]], # total masses
        [first_state_matrix[L_index][-1]] # total luminosities
    ])

    # iterate through every central temperature (after the initial
    # central temperature) and generate a star at each value
    for i, in_central_temp in enumerate(all_cen_temp[1:]):
        # progress indicator
        print("\nSolving Star {}".format(i + 2))

        # use the central temperature and central density arrays to
        # extrapolate a central density estimate for the current star
        rho_0_prediction = extrapolate(
            x=in_central_temp, # current central temperature
            x_arr=important_properties[2][:], # central temperature array
            y_arr=important_properties[1][:], # central density array
            calc_method="logpolyfit",
            degree=1 # use a decent polynomial fit degree parameter to avoid Rank Warnings
        )

        # use the predicted central density to generate the star
        __, radius_arr, state_matrix = solve_structure(
            stellar_structure,
            T_0=in_central_temp,
            rho_0_guess=rho_0_prediction,
            confidence=confidence
        )

        # create an array of the same shape as `important_properties` to contain the new data
        new_properties = numpy.array([
            [radius_arr[-1]], # surface radii
            [state_matrix[rho_index][0]], # central densities
            [state_matrix[T_index][0]], # central temperature
            [state_matrix[T_index][-1]], # surface temperatures
            [state_matrix[M_index][-1]], # total masses
            [state_matrix[L_index][-1]] # total luminosities
        ])

        # add the array of new properties to the array of all properties
        important_properties = numpy.concatenate((important_properties, new_properties), axis=1)

    # save the stellar sequence
    if save_data:
        Store().save(data=important_properties, data_filename="stellar_sequence.pickle")

    # get the surface temperature data
    surf_temp_data = important_properties[3][:]

    # get the luminosity data
    luminosity_data = important_properties[5][:]

    # use the stellar sequence to generate a Hertzsprung-Russell diagram
    hr_diagram(
        effect_temps=surf_temp_data,
        luminosities=luminosity_data
    )


def execute():
    """
    Function to execute a method of analysis as specified by the user.
    """

    # create a dictionary of all the supported functions
    all_functions = {
        "Generate a Star": [_generate_star, "0"],
        "Generate a Stellar Sequence": [_generate_star_sequence, "1"]
    }

    # print all the functions that are included in this module;
    # Also include a number used to select a function
    print("\nAll Available Functions:\n")
    for f_descr in all_functions.keys():
        print("[{}] {}".format(all_functions[f_descr][1], f_descr))

    # instruct the user to select a function from those listed
    f_num = input("\nSelect a Function Number >>> ")

    # execute the function corresponding to the selected function number
    for f_descr in all_functions.keys():
        if all_functions[f_descr][1] == f_num:
            print("\nExecuting '{}'...".format(f_descr))
            all_functions[f_descr][0]()


if __name__ == "__main__":
    execute()
