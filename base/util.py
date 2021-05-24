"""
Module to contain additional functions used throughout the environment.

:title: util.py

:author: Mitchell Shahen

:history: 10/05/2021
"""

import numpy
from scipy.optimize import curve_fit


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


def normalize_data(in_data=None, norm_values=None):
    """
    Function to normalize an input dataset. Handles multi-dimensional arrays by using values
    in the `norm_values` to normalize each dimensional array.

    :param in_data: A numpy ndarray of data to be normalized.
    :param norm_values: A list of values normalizing each of the dimensional arrays. Must match
        the size and order of the `in_data` parameter.
    :return: A tuple containing two numpy ndarrays: the normalized radii and the normalized states.
    """

    if in_data.shape[0] == 1:
        # scale the 1-Dimensional array
        in_data = numpy.array([item / norm_values[0] for item in in_data])
        in_data = in_data[0]

    else:
        # scale the multi-dimensional input array by each normalization value
        for i, dim_array in enumerate(in_data):
            in_data[i] = numpy.array([item / norm_values[i] for item in dim_array])

    return in_data


def extrapolate(x=None, x_arr=None, y_arr=None, calc_method="", degree=1):
    """
    Function for using various fitting calculations on a set of x values and y values to
    extrapolate a y-value corresponding to the inputted x value.

    :param x: The value at which the intended extrapolated value corresponds to.
    :param x_arr: An array of x values to be used in the extrapolation calculations.
    :param y_arr: An array of y values to be used in the extrapolation calculations.
    :param calc_method: A string representing the calculation method for generating an
        extrapolation prediction.
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
