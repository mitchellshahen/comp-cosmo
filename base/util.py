"""
Module to contain additional functions used throughout the environment.

:title: util.py

:author: Mitchell Shahen

:history: 10/05/2021
"""

from constants import L_sun, M_sun, r_sun, rho_0_sun, T_0_sun, T_sun
import numpy
from scipy.optimize import curve_fit
from stellar_structure import L_index, M_index, rho_index, T_index


def find_zeros(in_data=None, find_first=True):
    """
    Function to interpolate and find the values in an input dataset where the data flips from
    positive to negative or vice versa.

    :param in_data: A 1-Dimensional numpy array.
    :param find_first: A boolean indicating if only the first instance of a zero is outputted.
    :return: An array containing the indices, as floating point values, where sign flips occur.
    """

    # convert the data to True or False values depending on whether or not the data is negative
    bool_arr = numpy.signbit(in_data)

    # find the instances where False -> True or True -> False in the boolean array
    bool_flip_arr = numpy.diff(bool_arr)

    # find the indices where the flip array gives True (indicating a sign flip in `in_data`)
    sign_flip_arr = numpy.where(bool_flip_arr)

    all_zero_indices = []

    for sign_flip_index in sign_flip_arr:
        # get the total distance between the two values where the sign flip occurs
        total_distance = in_data[sign_flip_index + 1] - in_data[sign_flip_index]

        # find the exact floating point index of the zero value
        zero_index = sign_flip_index - (in_data[sign_flip_index] / total_distance)

        all_zero_indices.append(zero_index)

    # convert the list of zero indices to an array
    out_zeros = numpy.array(all_zero_indices)

    # if find first is True, cut the zeros array to just the first value
    if find_first:
        out_zeros = out_zeros[0]

    return out_zeros


def interpolate(in_data=None, index=0.0):
    """
    Interpolate the inputted dataset at the specified index.

    :param in_data: The array or multi-dimensional matrix to interpolate from.
    :param index: The index at which interpolation occurs.
    :return: The interpolated data point as a value (array input) or an array (matrix input).
    """

    # if the input dataset is a 1-Dimensional array, use numpy.interp to interpolate the data point
    if len(in_data.shape) == 1:
        index_arr = numpy.arange(len(in_data))
        interp_data = numpy.interp(index, index_arr, in_data)

    # if the input dataset is a matrix, use interp1d to interpolate the data
    # point along the second axis
    else:
        all_interp = []
        for arr in in_data:
            index_arr = numpy.arange(len(arr))
            intermed_interp = numpy.interp(index, index_arr, arr)[0]
            all_interp.append(intermed_interp)
        interp_data = numpy.array(all_interp)

    return interp_data


def normalize_data(r, state, radius_norm=1.0, state_norm=None):
    """
    Function to normalize the radius values and stellar state values using the inputted radius and
    stellar state normalization values.

    :param r: A numpy ndarray of radius values.
    :param state: A multi-dimensional matrix of stellar values including density, temperature,
        mass, luminosity, and optical depth.
    :param radius_norm: A value to normalize the radius values.
    :param state_norm: A list of values normalizing each of the state values. Must match the size
        and order of the `state` parameter.
    :return: A tuple containing two numpy ndarrays: the normalized radii and the normalized states.
    """

    # scale the 1-D radius values by the inputted radius norm
    r = numpy.array([item / radius_norm for item in r])

    # scale the multi-dimensional state by each inputted state norm value
    for i, state_arr in enumerate(state):
        state[i] = numpy.array([item / state_norm[i] for item in state_arr])

    return r, state


def extrapolate(x=None, x_arr=None, y_arr=None, calc_method="", degree=1):
    """
    Function for using various fitting calculations on a set of x values and y values to extrapolate a y-value
    corresponding to the inputted x value.

    :param x: The value at which the intended extrapolated value corresponds to.
    :param x_arr: An array of x values to be used in the extrapolation calculations.
    :param y_arr: An array of y values to be used in the extrapolation calculations.
    :param calc_method: A string representing the calculation method for generating an extrapolation prediction.
    :param degree: The polynomial degree that is calculated and used for the fitting calculations.
        If degree is `None`, an exponential fit is used instead.
    :return: The y value extrapolated to correspond with the x value parameter.
    """

    # if any of the inspected x value, x array, or y array inputs are invalid, return None
    if any(
        [
            x is None,
            len(x_arr) == 0,
            len(y_arr) == 0
        ]
    ):
        return None

    # ensure the inputted calculation type is supported
    supported_calc_types = ["polyfit", "logpolyfit", "expfit"]
    if calc_method not in supported_calc_types:
        raise IOError(
            "ERROR: Invalid Prediction Calculation Method. "
            "Supported Calculation Methods Include: {}".format(
                supported_calc_types
            )
        )

    # ensure the degree parameter is a non-zero, non-negative integer;
    # also acceptable is a degree parameter of None, but only when using the exponential fit method
    if not isinstance(degree, int):
        if degree is None:
            calc_method = "exp_fit"
        else:
            raise IOError("The degree parameter must be an integer.")
    elif degree <= 0:
        raise IOError("The degree parameter must be positive.")

    if calc_method == "polyfit":
        # get the polynomial fit valus(s)
        polynomial = numpy.polyfit(x_arr, y_arr, deg=degree)

        # use the polynomial fit value(s) to calculate the prediction
        prediction = numpy.poly1d(polynomial)(x)

    elif calc_method == "logpolyfit":
        # set the x-axis and y-axis datasets used to calculate the polynomial fit
        x_data = numpy.log(x_arr[-(degree + 1):])
        y_data = numpy.log(y_arr[-(degree + 1):])

        # calculate the polynomial fit
        polynomial = numpy.polyfit(x_data, y_data, deg=len(x_data) - 1)

        # use the polynomial fit value(s) to calculate the prediction
        prediction = numpy.exp(numpy.poly1d(polynomial)(numpy.log(x)))

    else:
        # set the exponential fit
        def exponential_fit(value, coeff=1, exp_coeff=1, const=1):
            return coeff * numpy.exp(exp_coeff * value) + const

        # calculate the fitting parameters
        fitting_parameters, __ = curve_fit(exponential_fit, x_arr, y_arr)
        a, b, c = fitting_parameters

        # use the exponential fit value(s) to calculate the prediction
        prediction = exponential_fit(x, coeff=a, exp_coeff=b, const=c)

    return prediction


def stellar_info(error, r, state, precision=3):
    """
    Function to print useful information pertaining to the stellar solution that generated the
    inputted radius array and stellar state.

    :param error: The luminosity error associated with the stellar solution that generated the
        radius array and state matrix.
    :param r: An array of radius values generated by solving the stellar structure equations.
    :param state: A matrix of state values generated by solving the stellar structure equations.
    :param precision: The precision used when printing the stellar information. Defaults to 3.
    """

    # set the precision of the printed values
    set_precision = ".{}E".format(precision)

    # print the relative luminosity error
    lumin_error = error[0]
    print("The relative error in luminosity is {}.".format(format(lumin_error, set_precision)))

    # format and print the surface radius in metres and in terms of the Sun's radius
    actual_surf_rad = r[-1]
    rel_surf_rad = r[-1] / r_sun
    print(
        "Surface Radius = {} metres or {} solar radii.".format(
            format(actual_surf_rad, set_precision), format(rel_surf_rad, set_precision)
        )
    )

    # format and print the central density in SI units & in terms of the Sun's central density
    actual_cen_density = state[rho_index][0]
    rel_cen_density = state[rho_index][0] / rho_0_sun
    print(
        "Core Density = {} kg/m^3 or {} times the Sun's core density.".format(
            format(actual_cen_density, set_precision), format(rel_cen_density, set_precision)
        )
    )

    # format and print the core temperature in Kelvin and in terms of the Sun's core temperature
    actual_cen_temp = state[T_index][0]
    rel_cen_temp = state[T_index][0] / T_0_sun
    print(
        "Core Temperature = {} Kelvin or {} times the Sun's core temperature.".format(
            format(actual_cen_temp, set_precision), format(rel_cen_temp, set_precision)
        )
    )

    # format & print the surface temperature in Kelvin and in terms of the Sun's surface temperature
    # Recall: the temperature was normalized to the solar core temperature
    actual_surf_temp = state[T_index][-1]
    rel_surf_temp = state[T_index][-1] / T_sun
    print(
        "Surface Temperature = {} Kelvin or {} times the Sun's surface temperature.".format(
            format(actual_surf_temp, set_precision), format(rel_surf_temp, set_precision)
        )
    )

    # format and print the total mass in kilograms and in terms of the solar mass
    actual_mass = state[M_index][-1]
    rel_mass = state[M_index][-1] / M_sun
    print(
        "Total Mass = {} kilograms or {} times the Sun's mass.".format(
            format(actual_mass, set_precision), format(rel_mass, set_precision)
        )
    )

    # format and print the total luminosity in Watts and in terms of the solar luminosity
    actual_lumin = state[L_index][-1]
    rel_lumin = state[L_index][-1] / L_sun
    print(
        "Total Luminosity = {} Watts of {} times the Sun's luminosity.".format(
            format(actual_lumin, set_precision), format(rel_lumin, set_precision)
        )
    )
