"""
Module to perform all the necessary functions and calculations to generate the datasets that
describe a star. Additionally, plots of the stellar structure variables are rendered.
"""

import numpy
from plot import hr_diagram, stellar_structure_plot
from solve_stellar import solve_structure
from stellar_structure import L_index, M_index, rho_index, T_index
from store import Store
import units
from util import predict_central_density, useful_info


def generate_star():
    """
    Function to perform all the necessary functions to solve and plot the stellar structure
    equations.
    """

    # set the central temperature
    # T_0 = 2e7 * units.K
    # T_0 = 8.23e6 * units.K
    T_0 = 1.5e7 * units.K

    # set the initial central density guess
    # rho_0_guess = 8.0063e4 * units.kg / (units.m ** 3)
    # rho_0_guess = 5.856e4 * units.kg / (units.m ** 3)
    rho_0_guess = 1e5 * units.kg / (units.m ** 3)

    # set the confidence
    confidence = 0.5

    # set the decision of whether or not to normalize the results
    normalize = False

    # set the decision of whether or not to save the data
    save_data = False

    # solve the stellar structure equations and acquire the necessary data
    lumin_error, radius_arr, state_matrix = solve_structure(
        T_0,
        rho_0_guess=rho_0_guess,
        confidence=confidence,
        normalize=normalize
    )

    # can only acquire and useful information about the stellar structure solutions
    # when the solution is not normalized
    if not normalize:
        useful_info(
            error=lumin_error,
            r=radius_arr,
            state=state_matrix,
            precision=3
        )

    # reformat the radius and add it to the state to form a single dataset for storage
    full_data = numpy.append(
        numpy.reshape(radius_arr, (1, -1)),
        state_matrix,
        axis=0
    )

    # save the outputted datasets for radius and state
    if save_data:
        if normalize:
            Store().save(data=full_data, data_filename="first_stellar_data.pickle")
        else:
            Store().save(data=full_data, data_filename="first_stellar_data_norm.pickle")

    # plot all the necessary graphs describing the above generated star only if it has been normalized
    if normalize:
        stellar_structure_plot(
            radius=full_data[0],
            density=full_data[rho_index + 1],
            temperature=full_data[T_index + 1],
            mass=full_data[M_index + 1],
            luminosity=full_data[L_index + 1]
        )


def generate_star_sequence():
    """
    Function to generate a star sequence: generating several stars and saving only each star's most
    important properties.
    """

    # number of stars to generate
    N = 10

    # initial central temperature to survey
    T_0_i = 1e5 * units.K

    # final central temperature to survey
    T_0_f = 1e8 * units.K

    # generate an array of temperature values to generate stars with
    all_cen_temp = numpy.linspace(T_0_i, T_0_f, N)

    # execute the solve_structure function for the first star using the default central density guess;
    # this is done to create non-empty arrays of central temperatures and central densities that can
    # be used to predict viable central densities for all the proceeding central temperature values
    print("\nSolving Star 1")
    __, first_radius_arr, first_state_matrix = solve_structure(
        T_0_i,
        rho_0_guess=1e4 * units.kg / (units.m ** 3),
        confidence=0.75,
        normalize=False
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

    # iterate through every central temperature (after the initial central temperature) generating a star at each value
    for i, in_central_temp in enumerate(all_cen_temp[1:]):
        # progress indicator
        print("\nSolving Star {}".format(i + 2))

        # use the central temperature and central density arrays to
        # calculate a central density estimate for the current star
        rho_0_prediction = predict_central_density(
            in_central_temp, # current central temperature
            important_properties[2][:], # central temperature array
            important_properties[1][:], # central density array
            calc_method="logpolyfit",
            degree=1 # use a decent polynomial fit degree parameter to avoid Rank Warnings
        )

        # use the predicted central density to generate the star with central temperature, `in_central_temp`
        __, radius_arr, state_matrix = solve_structure(
            in_central_temp,
            rho_0_guess=rho_0_prediction,
            confidence=0.5,
            normalize=False
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
    Store().save(data=important_properties, data_filename="first_stellar_sequence.pickle")

    # use the stellar sequence to generate a Hertzsprung-Russell diagram
    hr_diagram(
        effect_temps=important_properties[T_index][:],
        luminosities=important_properties[L_index][:]
    )


if __name__ == "__main__":
    generate_star()
