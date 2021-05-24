"""
Module to solve the stellar equations by means of a trial solution, testing the trial solution and
its parameters, and modifying the initial parameters to improve the next trial solution attempt.

:title: solve_stellar.py

:author: Mitchell Shahen

:history: 07/05/2021
"""

import numpy
from scipy import integrate
import sys
sys.path.append("../") # be able to access the base directory

from base.constants import M_sun, sigma
from base.stellar_structure import rho_index, T_index, M_index, L_index, tau_index
import base.units as units
from base.util import find_zeros, interpolate


def get_remaining_optical_depth(stellar_structure, r, state):
    """
    Method to obtain the remaining optical depth beyond a radius value. This remaining optical
    depth value is used to determine when the integration can be stopped.

    For R >> 1 and R >> r (like R ~ inf and r is finite):
        tau(R) - tau(r) ~= kappa * (rho ** 2) / abs(drho/dr)

    :param r: A radius value at which the remaining optical depth is to be calculated.
    :param state: An array containing the state values evaluated at the radius, `r`.
    :returns: The remaining optical depth value at the radius, `r`.
    """

    # extract the necessary values from the inputted state variable
    rho = state[rho_index]
    T = state[T_index]
    M = state[M_index]
    L = state[L_index]

    # calculate the mean opacity at the current density and temperature
    kappa = stellar_structure.mean_opacity(rho=rho, T=T)

    # calculate the radial derivative of the density at the current radius
    drho_dr = stellar_structure.hydrostat_equil(r=r, rho=rho, T=T, M=M, L=L)

    # calculate the remaining optical depth from the current radius to infinity
    remain_opt_depth = kappa * (rho ** 2) / numpy.abs(drho_dr)

    return remain_opt_depth


def test_luminosity(r, state):
    """
    Method to test if the calculated surface luminosity is consistent with the expected surface
    luminosity at the current temperature and density. The radius of the star's surface occurs
    at a value, r, such that the optical depth at infinity subtracted by the optical depth at r
    is equal to 2/3. Interpolation is used to determine the radius value where this is the case.

    The expected surface luminosity is calculated as:
        L_surf = 4 * pi * sigma * (r ** 2) * (T ** 4)

    :param r: An array of radius values from a non-zero minimum radius to (effectively) infinity.
    :param state: A matrix of state values evaluated at the various radii in `r`.
    :return: A tuple containing the luminosity error at the interpolated surface radius, an array of
        radius values truncated to the surface radius value, and a matrix of state values evaluated
        at each radius.
    """

    # get the optical depth at a very large radius (effectively infinity)
    tau_inf = state[tau_index, -1]

    # define the surface radius equation
    surf_rad_eq = tau_inf - state[tau_index, :] - (2 / 3)

    # find the exact index where the above equation equates to 0 (likely a floating point)
    surface_index = find_zeros(in_data=surf_rad_eq, find_first=True)

    # interpolate the radius array to find the radius value at the surface index
    surface_radius = interpolate(in_data=r, index=surface_index)

    # interpolate the state variable at the surface index to get the state at the surface
    surface_state = interpolate(in_data=state, index=surface_index)

    # calculate the expected luminosity at the surface using the surface radius and temperature
    expected_luminosity = 4 * numpy.pi * sigma * (
        surface_radius ** 2
    ) * (
        surface_state[T_index] ** 4
    )

    # calculate the error between the surface luminosity and the expected luminosity
    luminosity_error = surface_state[L_index] - expected_luminosity

    # normalize the luminosity error
    norm_constant = 1 / numpy.sqrt(surface_state[L_index] * expected_luminosity)
    luminosity_error *= norm_constant

    # set the temperature value to be consistent with the surface luminosity
    surface_state[T_index] = (
        surface_state[L_index] / (4 * numpy.pi * sigma * (surface_radius ** 2))
    ) ** (1/4)

    # limit the surface index to its next lowest integer
    surface_index = int(surface_index)

    # cut the radius array and state matrix to just prior to the surface index
    cut_r = r[:surface_index]
    cut_state = state[:, :surface_index]

    # add the surface data to the truncated radius array and state matrix
    trunc_r = numpy.append(cut_r, surface_radius)
    trunc_state = numpy.column_stack((cut_state, surface_state))

    return luminosity_error, trunc_r, trunc_state


def trial_solution(
        stellar_structure,
        r_0=1.0 * units.m,
        rho_0=1.0 * units.kg / (units.m ** 3),
        T_0=1.0 * units.K,
        optical_depth_threshold=1e-4,
        mass_threshold=1000 * M_sun,
        rtol=None,
        atol=None
):
    """
    Integrates the state of the star from an initial radius, r_0, until the estimated optical depth
    is below the specified threshold. The array of radius values and the state matrix are
    returned along with the fractional surface luminosity error.

    :param r_0: The initial radius. Must be greater than 0 to prevent numerical instabilities.
    :param rho_0: The density at a radius `r_0`.
    :param T_0: The temperature at a radius `r_0`.
    :param optical_depth_threshold: The remaining optical depth values at which the integration is
        allowed to stop.
    :param mass_threshold: The maximal stellar mass to prevent unbounded integration.
    :param rtol: The required relative accuracy during integration. Defaults to 1e-9.
    :param atol: The required absolute accuracy during integration. Defaults to rtol / 1000.
    :returns: The surface luminosity error and the array of radius value and the state matrix.
    """

    if rtol is None:
        rtol = 1e-9

    if atol is None:
        atol = rtol / 1000

    # define a function to determine when the integration has been completed to a sufficient degree
    def halt_integration(r, state):
        if state[M_index] > mass_threshold:
            return -1

        remaining_opt_depth = get_remaining_optical_depth(
            stellar_structure=stellar_structure,
            r=r,
            state=state
        )

        optical_depth_diff = remaining_opt_depth - optical_depth_threshold

        return optical_depth_diff

    halt_integration.terminal = True

    # Ending radius is infinity, integration will only be halted via the halt_integration event
    # Not sure what good values for atol and rtol are, but these seem to work well
    result = integrate.solve_ivp(
        stellar_structure.get_derivative_state,
        (r_0, numpy.inf),
        stellar_structure.initial_properties(r_0=r_0, rho_0=rho_0, T_0=T_0),
        events=halt_integration,
        atol=atol,
        rtol=rtol
    )

    # noinspection PyUnresolvedReferences
    r_values, state_values = result.t, result.y

    # acquire the luminosity error as well as the truncated radius array and state matrix
    luminosity_error, trunc_r, trunc_state = test_luminosity(r=r_values, state=state_values)

    return luminosity_error, trunc_r, trunc_state


def bisection_method(
        stellar_structure, T_0, rho_0_guess, confidence, rho_0_min, rho_0_max, rho_0_tol, rtol, optical_depth_threshold
):
    """
    Method to execute the bisection method with the notion of a confidence value.

    :param T_0: The initial radius. Must be greater than 0 to prevent numerical instabilities.
    :param rho_0_guess: The density at a radius `r_0`.
    :param confidence: The confidence of the central density guess. This parameter must be strictly
        between 0.5 (no confidence) and 1 (complete confidence).
    :param rho_0_min: The minimal central density guess.
    :param rho_0_max: The maximal central density guess.
    :param rho_0_tol: The tolerance associated with selecting the central density guess.
    :param rtol: The required relative accuracy during integration.
    :param optical_depth_threshold: The remaining optical depth value at which the integration is
        allowed to stop.
    :return: The optimal central density guess.
    """

    # set up an output central density and luminosity error to contain the last valid central density estimate and
    # its corresponding luminosity error; if an error occurs, return the last valid central density estimate
    output_rho_0 = 0
    L_error_guess = 0

    # try to perform the necessary bisection method calculations for as long as possible until
    # convergence is met or an error occurs
    try:
        # ---------- # ESTABLISH BIASES AND CHECK FOR IDEAL SOLUTIONS # ---------- #

        # calculate the luminosity error associated with the central density guess
        L_error_guess, __, __ = trial_solution(
            stellar_structure=stellar_structure,
            rho_0=rho_0_guess,
            T_0=T_0,
            optical_depth_threshold=optical_depth_threshold,
            rtol=rtol
        )

        # upon completing the initial trial solution successfully, replace the central density and luminosity error
        output_rho_0 = rho_0_guess
        L_error_guess = L_error_guess

        # if using the input central density guess resulted in no luminosity error, return the
        # luminosity error, radius values, and state values from using the density guess
        if L_error_guess == 0:
            return rho_0_guess

        # calculate the luminosity error associated with the minimum central density
        L_error_min, __, __ = trial_solution(
            stellar_structure=stellar_structure,
            rho_0=rho_0_min,
            T_0=T_0,
            optical_depth_threshold=optical_depth_threshold,
            rtol=rtol
        )

        # if using the minimum central density resulted in no luminosity error, return the
        # luminosity error, radius values, and state values from using the minimum density
        if L_error_min == 0:
            return rho_0_min

        # if the intended zero luminosity error is between the minimum and guess densities, set the
        # lower and upper central densities as the minimum and guess densities and set the biases
        if L_error_min < 0 < L_error_guess:
            rho_0_low, rho_0_high = rho_0_min, rho_0_guess
            bias_high = True
            bias_low = False
        elif L_error_guess < 0 < L_error_min:
            rho_0_low, rho_0_high = rho_0_guess, rho_0_min
            bias_low = True
            bias_high = False

        # if the luminosity error is not within the minimum and guessed densities, repeat the above
        # processes using the guess density and the maximum density
        else:
            # calculate the luminosity error associated with the maximum central density
            L_error_max, __, __ = trial_solution(
                stellar_structure=stellar_structure,
                rho_0=rho_0_max,
                T_0=T_0,
                optical_depth_threshold=optical_depth_threshold,
                rtol=rtol
            )

            # if using the maximum central density resulted in no luminosity error, return the
            # luminosity error, radius values, and state values from using the maximum density
            if L_error_max == 0:
                return rho_0_max

            # if the intended zero luminosity error is between the maximum and guess densities, set the
            # lower and upper central densities as the maximum and guess densities and set the biases
            if L_error_max < 0 < L_error_guess:
                rho_0_low, rho_0_high = rho_0_max, rho_0_guess
                bias_high = True
                bias_low = False
            elif L_error_guess < 0 < L_error_max:
                rho_0_low, rho_0_high = rho_0_guess, rho_0_max
                bias_low = True
                bias_high = False
            else:
                # if the central density that gives no error lies outside the density constraints,
                # re-run this function using larger density contraints
                print(
                    "Retrying with larger central density interval for a central temperature "
                    "of {} Kelvin".format(T_0)
                )

                # set confidence to be much higher since we know that the other
                # boundary will be even further from the guess
                return solve_structure(
                    stellar_structure,
                    T_0,
                    rho_0_guess=rho_0_guess,
                    confidence=(confidence + 4) / 5,
                    rho_0_min=rho_0_min / 1000,
                    rho_0_max=1000 * rho_0_max,
                    rho_0_tol=rho_0_tol,
                    rtol=rtol,
                    optical_depth_threshold=optical_depth_threshold
                )

        # ---------- # USE THE BIASES AND CONFIDENCE TO IMPROVE THE CENTRAL DENSITY GUESS # ---------- #

        # continue to increase the precision until the central density tolerance is met
        while numpy.abs(rho_0_high - rho_0_low) / 2 > rho_0_tol:
            # if the bias is favouring the lower bound, assign the central density guess using the upper
            # and lower density bounds with the confidence assigned to the lower density bound
            if bias_low:
                # assign the confidence to the lower density bound
                rho_0_guess = confidence * rho_0_low + (1 - confidence) * rho_0_high

                # check for numerical precision
                if rho_0_guess == rho_0_low or rho_0_guess == rho_0_high:
                    print('Reached limits of numerical precision for rho_0 using low bias')
                    break

                # calculate the luminosity error associated with the new central density guess
                L_error_guess, __, __ = trial_solution(
                    stellar_structure=stellar_structure,
                    rho_0=rho_0_guess,
                    T_0=T_0,
                    optical_depth_threshold=optical_depth_threshold,
                    rtol=rtol
                )

                # upon completing the bias low trial solution successfully,
                # replace the central density and luminosity error
                output_rho_0 = rho_0_guess
                L_error_guess = L_error_guess

                # check the luminosity error of the new central density guess and assign the
                # upper and lower densities accordingly
                if L_error_guess == 0:
                    return rho_0_guess
                if L_error_guess < 0:
                    rho_0_low = rho_0_guess
                    bias_low = False  # ignore initial guess bias since it is no longer the low endpoint
                elif L_error_guess > 0:
                    rho_0_high = rho_0_guess

            # if the bias is favouring the upper bound, assign the central density guess using the upper
            # and lower density bounds with the confidence assigned to the upper density bound
            elif bias_high:
                # assign the confidence to the upper density bound
                rho_0_guess = (1 - confidence) * rho_0_low + confidence * rho_0_high

                # check the numerical precision
                if rho_0_guess == rho_0_low or rho_0_guess == rho_0_high:
                    print('Reached limits of numerical precision for rho_0 using high bias')
                    break

                # calculate the luminosity error associated with the new central density guess
                L_error_guess, __, __ = trial_solution(
                    stellar_structure=stellar_structure,
                    rho_0=rho_0_guess,
                    T_0=T_0,
                    optical_depth_threshold=optical_depth_threshold,
                    rtol=rtol
                )

                # upon completing the bias high trial solution successfully,
                # replace the central density and luminosity error
                output_rho_0 = rho_0_guess
                L_error_guess = L_error_guess

                # check the luminosity error of the new central density guess and assign the
                # upper and lower densities accordingly
                if L_error_guess == 0:
                    return rho_0_guess
                if L_error_guess < 0:
                    rho_0_low = rho_0_guess
                elif L_error_guess > 0:
                    rho_0_high = rho_0_guess
                    bias_high = False # ignore initial guess bias since it's no longer the high endpoint

            # if the re-calculated luminosity error has flipped sign in a previous iteration, modify
            # the central density guess to be the midpoint between the upper and lower bounds
            else:
                # assign the central density guess as the midpoint between the upper and lower bounds
                rho_0_guess = (rho_0_low + rho_0_high) / 2

                # check the numerical precision
                if rho_0_guess == rho_0_low or rho_0_guess == rho_0_high:
                    print('Reached limits of numerical precision for rho_0 using midpoint')
                    break

                # calculate the luminosity error associated with the new central density guess
                L_error_guess, __, __ = trial_solution(
                    stellar_structure=stellar_structure,
                    rho_0=rho_0_guess,
                    T_0=T_0,
                    optical_depth_threshold=optical_depth_threshold,
                    rtol=rtol
                )

                # upon completing the midpoint trial solution successfully,
                # replace the central density and luminosity error
                output_rho_0 = rho_0_guess
                L_error_guess = L_error_guess

                # check the luminosity error of the new central density guess and assign the
                # upper and lower densities accordingly
                if L_error_guess == 0:
                    return rho_0_guess
                if L_error_guess < 0:
                    rho_0_low = rho_0_guess
                elif L_error_guess > 0:
                    rho_0_high = rho_0_guess

        # if no ideal error was obtained, output the midpoint between the upper and lower density bounds
        output_rho_0 = (rho_0_low + rho_0_high) / 2

    except ArithmeticError as error_message:
        # print that an error occurred forcing the last used valid central density was used
        print("Attempting to use a central density guess, {}, resulted in the following error:".format(output_rho_0))
        print(error_message)

        # check that at least one central density solution calculation executed without error,
        # if not re-run this function with a modified initial central density
        if any(
            [
                output_rho_0 == 0,
                L_error_guess == 0
            ]
        ):
            print(
                "The above error occurred when attempting to use the initial central density guess. The erroneous "
                "central density guess will be modified by 5% and the bisection method will be executed again with "
                "the modified central density guess and a lower confidence value."
            )

            # set the new, modified central density guess by varying the original guess by 5%
            mod_rho_0_guess = 1.05 * rho_0_guess

            # additionally, lower the confidence
            mod_confidence = confidence - ((confidence - 0.5) / 4)

            # re-run the bisection method
            return bisection_method(
                stellar_structure,
                T_0,
                mod_rho_0_guess,
                mod_confidence,
                rho_0_min,
                rho_0_max,
                rho_0_tol,
                rtol,
                optical_depth_threshold
            )
        else:
            print("The last valid central density guess will be used to generate the stellar equation solution.")

    return output_rho_0, L_error_guess


def solve_structure(
        stellar_structure,
        T_0=1e6 * (units.K),
        rho_0_guess=1e5 * (units.kg / (units.m ** 3)),
        confidence=0.9,
        rho_0_min=300 * (units.kg / (units.m ** 3)),
        rho_0_max=4e9 * (units.kg / (units.m ** 3)),
        rho_0_tol=1e-20 * (units.kg / (units.m ** 3)),
        rtol=1e-11,
        optical_depth_threshold=1e-4
):
    """
    Solves the stellar structure equations in `stellar_structure.py` using the inputted central
    temperature, `T_0`, by means of the point-and-shoot method. Also employed is the bisection
    algorithm modified with the notion of confidence. A higher confidence results in a faster
    convergence for the central density guess, `rho_0_guess`. Once rho_0_guess falls outside the
    interval of interest, simple bisection is used. Too low of a confidence will cause this to
    reduce to simple bisection, and too high of a confidence will likely cause rho_0_guess to fall
    outside the range of interest too fast leaving an unnecessarily large remaining search space.

    :param T_0: The central temperature.
    :param rho_0_guess: A guess for the central density.
    :param confidence: The confidence of the guess. This parameter must be strictly between 0.5 (no
        confidence) and 1 (complete confidence).
    :param rho_0_min: The minimum possible central density.
    :param rho_0_max: The maximum possible central density.
    :param rho_0_tol: The tolerance within which the central density must be determined for
        integration to be concluded.
    :param rtol: The required relative accuracy during integration.
    :param optical_depth_threshold: The remaining optical depth values at which the integration is
        allowed to stop.
    :return: The resulting fractional luminosity error, r_values, and state_values of the resulting
        stellar structure solution.
    """

    # if the input central density guess is None, use the default
    if rho_0_guess is None:
        rho_0_guess = 1e5 * (units.kg / (units.m ** 3))

    # ensure the confidence parameter is properly specified
    if confidence < 0.5:
        raise IOError("Confidence must be at least 0.5!")
    if confidence >= 1:
        raise IOError("Confidence must be less than 1!")

    rho_0, L_error_guess = bisection_method(
        stellar_structure,
        T_0,
        rho_0_guess,
        confidence,
        rho_0_min,
        rho_0_max,
        rho_0_tol,
        rtol,
        optical_depth_threshold
    )

    # using the final central density value, generate the final stellar structure solution
    final_error, final_r, final_state = trial_solution(
        stellar_structure=stellar_structure,
        rho_0=rho_0,
        T_0=T_0,
        optical_depth_threshold=optical_depth_threshold,
        rtol=rtol
    )

    # double check the final error
    if numpy.abs(final_error) > 1000:
        # increase the confidence in the central density guess
        higher_accuracy = confidence + ((1.0 - confidence) / 4)
        print(
            "Solution failed to converge. Retrying stellar equation solution for central temperature, "
            "T = {} Kelvin, with higher accuracy: {} --> {}".format(T_0, confidence, higher_accuracy)
        )
        return solve_structure(
            stellar_structure,
            T_0=T_0,
            rho_0_guess=rho_0,
            confidence=higher_accuracy, # set the confidence to be much higher now
            rho_0_min=rho_0_min,
            rho_0_max=rho_0_max,
            rho_0_tol=rho_0_tol,
            rtol=rtol * 100,
            optical_depth_threshold=optical_depth_threshold
        )

    return final_error, final_r, final_state
