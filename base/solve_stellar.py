"""
Module to solve the stellar equations by means of a trial solution, testing the trial solution, and
modifying the initial parameters to improve the next trial solution attempt.
"""

from constants import M_sun, sigma
import numpy
from scipy import integrate
from stellar_structure import StellarStructure, T_index, M_index, L_index, tau_index
import units
from util import find_zeros, interpolate


def get_remaining_optical_depth(r, state):
    """
    Method to obtain the remaining optical depth beyond a radius value. This remaining optical
    depth value is used to determine when the integration can be stopped.

    For R >> 1 and R >> r:
        tau(R) - tau(r) ~= kappa * (rho ** 2) / abs(drho/dr)
    """

    # extract the necessary values from the inputted state variable
    rho, T, M, L, tau = state

    # calculate the mean opacity at the current density and temperature
    kappa = StellarStructure().mean_opacity(rho=rho, T=T)

    # calculate the radial derivative of the density at the current radius
    drho_dr = StellarStructure().hydrostat_equil(r=r, rho=rho, T=T, M=M, L=L)

    # calculate the remaining optical depth from the current radius to infinity
    remain_opt_depth = kappa * (rho ** 2) / abs(drho_dr)

    return remain_opt_depth


def test_luminosity(r, state):
    """
    Method to test if the calculated surface luminosity is consistent with the expected surface
    luminosity at the current temperature and density. The radius of the star's surface occurs
    at a value, r, such that the optical depth at infinity subtracted by the optical depth at r
    is equal to 2/3. Interpolation is used to determine the radius value where this is the case.

    The expected surface luminosity is calculated as:
        L_surf = 4 * pi * sigma * (r ** 2) * (T ** 4)

    :return:
    """

    # get the optical depth at a very large radius (effectively infinity)
    tau_inf = state[tau_index, -1]

    # define the surface radius equation
    surf_rad_eq = tau_inf - state[tau_index, :] - (2 / 3)

    # find the exact index where the above equation equates to 0 (likely a floating point)
    surface_index = find_zeros(in_data=surf_rad_eq)

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

    # cut the radius array and state values to just before the surface and add the surface data
    r = numpy.append(r[:surface_index], surface_radius)
    state = numpy.column_stack((state[:, :surface_index], surface_state))

    return luminosity_error, r, state


def trial_solution(
        r_0=1.0 * units.m,
        rho_0=1.0 * units.kg / (units.m ** 3),
        T_0=1.0 * units.K,
        optical_depth_threshold=1e-4,
        mass_threshold=1000 * M_sun,
        rtol=1e-9,
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
    :param rtol: The required relative accuracy during integration.
    :param atol: The required absolute accuracy during integration. Defaults to rtol / 1000.
    :returns: The surface luminosity error and the array of radius values and the state matrix.
    """

    if atol is None:
        atol = rtol / 1000

    # define a function to determine when the integration has been completed to a sufficient degree
    def halt_integration(r, state):
        if state[M_index] > mass_threshold:
            return -1
        return get_remaining_optical_depth(r=r, state=state) - optical_depth_threshold

    halt_integration.terminal = True

    # Ending radius is infinity, integration will only be halted via the halt_integration event
    # Not sure what good values for atol and rtol are, but these seem to work well
    result = integrate.solve_ivp(
        StellarStructure().get_derivative_state,
        (r_0, numpy.inf),
        StellarStructure().initial_properties(r_0=r_0, rho_0=rho_0, T_0=T_0),
        events=halt_integration,
        atol=atol,
        rtol=rtol
    )

    # noinspection PyUnresolvedReferences
    r_values, state_values = result.t, result.y

    return test_luminosity(r=r_values, state=state_values)


def solve_structure(
        T_0,
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

    if confidence < 0.5:
        raise IOError("Confidence must be at least 0.5!")
    if confidence >= 1:
        raise IOError("Confidence must be less than 1!")

    # calculate the luminosity error associated with the central density guess
    L_error_guess, __, __ = trial_solution(
        rho_0=rho_0_guess,
        T_0=T_0,
        optical_depth_threshold=optical_depth_threshold,
        rtol=rtol
    )

    # if using the input central density guess resulted in no
    # luminosity error, return the density guess
    if L_error_guess == 0:
        return rho_0_guess

    # calculate the luminosity error associated with the minimum central density
    L_error_min, __, __ = trial_solution(
        rho_0=rho_0_min,
        T_0=T_0,
        optical_depth_threshold=optical_depth_threshold,
        rtol=rtol
    )

    # if using the minimum central density guess resulted in no
    # luminosity error, return the minimum central density
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
            rho_0=rho_0_max,
            T_0=T_0,
            optical_depth_threshold=optical_depth_threshold,
            rtol=rtol
        )

        # if using the maximum central density guess resulted in no
        # luminosity error, return the maximum central density
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
                T_0,
                rho_0_guess=rho_0_guess,
                confidence=(confidence + 4) / 5,
                rho_0_min=rho_0_min / 1000,
                rho_0_max=1000 * rho_0_max,
                rho_0_tol=rho_0_tol,
                rtol=rtol,
                optical_depth_threshold=optical_depth_threshold
            )

    # continue to increase the precision until the central density tolerance is met
    while numpy.abs(rho_0_high - rho_0_low) / 2 > rho_0_tol:
        # if the bias is favouring the lower bound, scale the central density guess using the
        # confidence assigned to the lower density bound
        if bias_low:
            # assign the confidence to the lower density bound
            rho_0_guess = confidence * rho_0_low + (1 - confidence) * rho_0_high

            # check for numerical precision
            if rho_0_guess == rho_0_low or rho_0_guess == rho_0_high:
                print('Reached limits of numerical precision for rho_0')
                break

            # calculate the luminosity error associated with the new central density guess
            L_error_guess, __, __ = trial_solution(
                rho_0=rho_0_guess,
                T_0=T_0,
                optical_depth_threshold=optical_depth_threshold,
                rtol=rtol
            )

            # check the luminosity error of the new central density guess
            if L_error_guess == 0:
                return rho_0_guess
            if L_error_guess < 0:
                rho_0_low = rho_0_guess
                bias_low = False # ignore initial guess bias since it is no longer the low endpoint
            elif L_error_guess > 0:
                rho_0_high = rho_0_guess

        # if the bias is favouring the upper bound, scale the central density guess using the
        # confidence assigned to the upper density bound
        elif bias_high:
            # assign the confidence to the upper density bound
            rho_0_guess = (1 - confidence) * rho_0_low + confidence * rho_0_high

            # check the numerical precision
            if rho_0_guess == rho_0_low or rho_0_guess == rho_0_high:
                print('Reached limits of numerical precision for rho_0')
                break

            # calculate the luminosity error associated with the new central density guess
            L_error_guess, __, __ = trial_solution(
                rho_0=rho_0_guess,
                T_0=T_0,
                optical_depth_threshold=optical_depth_threshold,
                rtol=rtol
            )

            # check the luminosity error of the new central density guess
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
                print('Reached limits of numerical precision for rho_0')
                break

            # calculate the luminosity error associated with the new central density guess
            L_error_guess, __, __ = trial_solution(
                rho_0=rho_0_guess,
                T_0=T_0,
                optical_depth_threshold=optical_depth_threshold,
                rtol=rtol
            )

            # check the luminosity error of the new central density guess
            if L_error_guess == 0:
                return rho_0_guess
            if L_error_guess < 0:
                rho_0_low = rho_0_guess
            elif L_error_guess > 0:
                rho_0_high = rho_0_guess

    # assign the final central density as the midpoint between the upper and lower density bounds
    rho_0 = (rho_0_high + rho_0_low) / 2

    # if solution failed to converge, recurse with greater accuracy
    if numpy.abs(L_error_guess) > 1000:
        print(
            "Retrying stellar equation solution for central temperature, T = {} Kelvin, "
            "with higher accuracy: {} --> {}".format(T_0, confidence, 0.99)
        )
        return solve_structure(
            T_0,
            rho_0_guess=rho_0,
            confidence=0.99,  # set the confidence to be extremely high now
            rho_0_min=rho_0_min,
            rho_0_max=rho_0_max,
            rho_0_tol=rho_0_tol,
            rtol=rtol * 100,
            optical_depth_threshold=optical_depth_threshold
        )

    # Generate the final stellar structure solution
    final_solution = trial_solution(
        rho_0=rho_0,
        T_0=T_0,
        optical_depth_threshold=optical_depth_threshold,
        rtol=rtol
    )

    return final_solution
