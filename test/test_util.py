"""
Module to test the function in the `base/util.py` file.
"""

import numpy
import sys
import unittest

sys.path.append("../")

from base import util


class UtilTestCase(unittest.TestCase):
    """
    Unit testing class for testing the various functions of the `base/util.py` module.
    """

    # ---------- # TESTING EXTRAPOLATE FUNCTION # ---------- #

    def test_extrapolate_polyfit(self, silent=True):
        """
        Test the extrapolate function's ability to accurately predict the next value of a function
        using the polynomial fit calculation method.
        """

        # set up the function for which the extrapolation method is to be tested
        def func(x):
            return 2 * x**2 + 15 * x + 4

        # set up the linear dataset
        x_data = numpy.arange(0, 10, 1)
        y_data = func(x_data)

        # define the next value and its index in the array
        next_value_index = x_data[-1] + 1
        next_value = func(next_value_index)

        # extrapolate the next value using each supported calculation method
        prediction_polyfit = util.extrapolate(
            x=next_value_index,
            x_arr=x_data,
            y_arr=y_data,
            calc_method="polyfit",
            degree=2
        )

        # determine the error in each prediction method
        polyfit_error = abs(1.0 - prediction_polyfit / next_value)

        # check that the relative error is sufficiently close to 0
        self.assertAlmostEqual(
            polyfit_error,
            0,
            places=7,
            msg="Using a polynomial fit with degree 2, the "
                "relative error exceeds that which is allowed."
        )

        if not silent:
            # print the results
            print(
                "Using a polynomial fit of degree 2, the extrapolation function predicts the "
                "executed function's behaviour with a relative error of {} %".format(
                    format(polyfit_error * 100, ".2E")
                )
            )

    def test_extrapolate_logpolyfit(self, silent=True):
        """
        Test the extrapolate function's ability to accurately predict the next value of a function
        using the logarithmic polynomial fit calculation method.
        """

        # set up the function for which the extrapolation method is to be tested
        def func(x):
            return 2 * x**2 + 15 * x + 4

        # set up the linear dataset
        x_data = numpy.arange(0, 10, 1)
        y_data = func(x_data)

        # define the next value and its index in the array
        next_value_index = x_data[-1] + 1
        next_value = func(next_value_index)

        # extrapolate the next value using each supported calculation method
        prediction_logpolyfit = util.extrapolate(
            x=next_value_index,
            x_arr=x_data,
            y_arr=y_data,
            calc_method="logpolyfit",
            degree=3
        )

        # determine the error in each prediction method
        logpolyfit_error = abs(1.0 - prediction_logpolyfit / next_value)

        # check that the relative error is sufficiently close to 0
        self.assertAlmostEqual(
            logpolyfit_error,
            0,
            places=4,
            msg="Using a logarithmic polynomial fit with degree 2, the "
                "relative error exceeds that which is allowed."
        )

        if not silent:
            # print the results
            print(
                "Using a logarithmic polynomial fit of degree 2, the extrapolation function "
                "predicts the executed function's behaviour to within an error of {} %".format(
                    format(logpolyfit_error * 100, ".2E")
                )
            )

    # ---------- # TESTING FIND ZEROS FUNCTION # ---------- #

    def test_find_zeros_no_data(self):
        """
        Test that the `find_zeros` function can handle receiving an empty dataset by returning an
        empty array or returning `None` (if the `find_first` parameter is True).
        """

        # set the empty input data array
        input_data = numpy.array([])

        # perform the find zeros function with and without `find_first`
        out_zeros_first = util.find_zeros(in_data=input_data, find_first=True)
        out_zeros_all = util.find_zeros(in_data=input_data, find_first=False)

        # validate that the resulting values are `None` and an empty array, respectively
        self.assertIsNone(out_zeros_first)
        self.assertTrue(out_zeros_all.shape[1] == 0)

    def test_find_zeros_positive_data(self):
        """
        Test that the `find_zeros` function can receive a dataset of strictly positive values and
        return an empty array or `None` (if the `find_first` parameter is True) as there are no
        zeros to be found.
        """

        # set up the all positive input dataset
        input_data = numpy.arange(1, 100, 1)

        # perform the find zeros function with and without `find_first`
        out_zeros_first = util.find_zeros(in_data=input_data, find_first=True)
        out_zeros_all = util.find_zeros(in_data=input_data, find_first=False)

        # validate that the resulting values are `None` and an empty array, respectively
        self.assertIsNone(out_zeros_first)
        self.assertTrue(out_zeros_all.shape[1] == 0)

    # ---------- # TESTING INTERPOLATE FUNCTION # ---------- #

    def test_interpolate(self):
        return None

    # ---------- # TESTING NORMALIZE DATA FUNCTION # ---------- #

    def test_normalize_data(self):
        return None


UtilTestCase().test_extrapolate_logpolyfit(silent=False)
