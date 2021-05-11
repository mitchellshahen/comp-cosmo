"""
Module to house additional functions.
"""

import numpy
from scipy.interpolate import interp1d


def find_zeros(in_data=None):
    """
    Function to interpolate and find the values in an input dataset where the data flips from
    positive to negative or vice versa.

    :param in_data: A 1-Dimensional numpy array.
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
    all_zeros = numpy.array(all_zero_indices)

    return all_zeros


def interpolate(in_data=None, index=0.0):
    """
    Interpolate the inputted dataset at the specified index.

    :param in_data: The array or 2D matrix to interpolate from.
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
        index_arr = numpy.arange(in_data.shape[1])
        interp_data = interp1d(index_arr, in_data)(index)

    return interp_data
