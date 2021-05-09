"""
Module for Testing of the Numerical Integration module: `base.num_integ.NumericalIntegration`
"""

# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=wrong-import-position

import numpy
import sys
import unittest

# access the comp-cosmo directory
sys.path.append("../../../")

from base.num_integ import NumericalIntegration


# ---------- # VALID NUMERICAL INTEGRATION CLASS FUNCTIONS # ---------- #


class ValidExpFunc:
    def __init__(self, a):
        self.a = a

    def function(self, x, y):
        return numpy.array(-1.0 * self.a * y)


class ValidHarmOscill:
    def __init__(self, w):
        self.w = w

    def function(self, x, y):
        return numpy.array([y[1], -1 * (self.w ** 2) * y[0]])


# ---------- # VALID NUMERICAL INTEGRATION CLASS FUNCTIONS # ---------- #


class InvalidFunc:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def not_a_function(self, x, y):
        return numpy.array(self.a + self.b + x + y)


class NumericalIntegrationTestCase(unittest.TestCase):
    """
    Unit testing class for the NumericalIntegration class.
    """

    def test_validate_parameters(self):
        """
        Method to test the validate parameters functionality.
        """

        # ---------- # VALID NUMERICAL INTEGRATION PARAMETERS # ---------- #

        # valid numerical integration technique
        valid_technique = "eulers_method"

        # valid range of independent variable values
        valid_x_range = [0.0, 1.0]

        # valid non-zero step value
        valid_step_value = 1.0

        # valid initial dependent variable value(s)
        valid_y_initial = [0.0]

        # ---------- # INVALID NUMERICAL INTEGRATION PARAMETERS # ---------- #

        # valid numerical integration technique
        invalid_technique = "nothing"

        # valid range of independent variable values
        invalid_x_range = []

        # valid non-zero step value
        invalid_step_value = 0.0

        # valid initial dependent variable value(s)
        invalid_y_initial = ["hello", "world"]

        # ---------- # NUMERICAL INTEGRATION UNIT TESTING # ---------- #

        # test that all valid parameters will cause the validate function to allow proceeding
        proceed_all_valid = NumericalIntegration(technique=valid_technique)._validate(
            f_class=ValidExpFunc(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertTrue(proceed_all_valid)

        # test that the validate function will notice an erroneous integration technique
        proceed_bad_technique = NumericalIntegration(technique=invalid_technique)._validate(
            f_class=ValidExpFunc(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_technique)

        # test that the validate function will notice an erroneous function class
        proceed_bad_f_class = NumericalIntegration(technique=valid_technique)._validate(
            f_class=InvalidFunc(a=0.0, b=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_f_class)

        # test that the validate function will notice an erroneous x_range list
        proceed_bad_x_range = NumericalIntegration(technique=valid_technique)._validate(
            f_class=ValidExpFunc(a=0.0),
            x_range=invalid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_x_range)

        # test that the validate function will notice an erroneous step value input
        proceed_bad_step_value = NumericalIntegration(technique=valid_technique)._validate(
            f_class=ValidExpFunc(a=0.0),
            x_range=valid_x_range,
            step_value=invalid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_step_value)

        # test that the validate function will notice an erroneous initial y_value list
        proceed_bad_y_initial = NumericalIntegration(technique=valid_technique)._validate(
            f_class=ValidExpFunc(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=invalid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_y_initial)

    def test_integration_exponential(self, silent=True):
        """
        Unit test to test that each integration method approximately solves an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = [10.0]

        # acquire all the supported integration techniques
        all_techniques = list(NumericalIntegration().supported_techniques.keys())

        # iterate through each integration technique testing each gives a valid approximation
        for integration_technique in all_techniques:
            x_arr, y_arr = NumericalIntegration(technique=integration_technique).execute(
                f_class=ValidExpFunc(a=differential_coeff),
                x_range=[0.0, 10.0],
                step_value=0.001,
                y_initial=initial_value,
                silent=silent
            )
            self.assertIsNotNone(
                x_arr,
                msg="Independent values array is None as the integration could not be performed."
            )
            self.assertIsNotNone(
                y_arr,
                msg="Dependent values array is None as the integration could not be performed."
            )

            # set up the validated x and y arrays using the known value of the integrated solution
            y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

            # set up variables to track convergence failures and the allowed error
            convergence_failures = False
            maximal_error = 1.0

            # iterate and find the maximal allowed error before encountering convergence failures
            while not convergence_failures:
                for i, __ in enumerate(x_arr):
                    # obtain the calculated value at the current x_value
                    y_calc = y_arr[0][i]

                    # obtain the exact value at the current x_value
                    y_known = y_solution[i]

                    # calculate the relative y_value to be compared with a value of 1
                    # Note: A relative y value of 1 indicates perfect convergence
                    relative_y = y_calc / y_known

                    # compare the relative y_value to perfect convergence
                    if abs(relative_y - 1) >= maximal_error:
                        convergence_failures = True
                        break

                # if not convergence failures are found, increase the precision
                if not convergence_failures:
                    maximal_error /= 10

            if not silent:
                # print a message including the maximal error for each integration technique
                print(
                    "Estimating the exact solution with {} using {} steps yielded no "
                    "convergence failures up to a maximal error of {}.".format(
                        integration_technique,
                        len(x_arr),
                        maximal_error
                    )
                )

    def test_integration_harm_oscill(self, silent=True):
        """
        Test to test that each integration method approximately solves a simple harmonic oscillator:
            d2y/dx2 = -1.0 * (w ** 2) * y
        The above harmonic oscillator yields the following form when integrated:
            y(x) = y(0) * cos(w * x) + dy(0)/dx * sin(w * x) / w
        We will set y(0) = 2, dy(0)/dx = 1.0, w = 2.0, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        w = 2.0
        initial_y = 2.0
        initial_dy_dx = 0.0
        x_range = [0.0, 10.0]

        # acquire all the supported integration techniques
        all_techniques = list(NumericalIntegration().supported_techniques.keys())

        # iterate through each integration technique testing each performs the intended integration
        for integration_technique in all_techniques:
            x_arr, y_arr = NumericalIntegration(technique=integration_technique).execute(
                f_class=ValidHarmOscill(w=w),
                x_range=x_range,
                step_value=0.0001,
                y_initial=[initial_y, initial_dy_dx],
                silent=silent
            )
            self.assertIsNotNone(
                x_arr,
                msg="Independent values array is None as the integration could not be performed."
            )
            self.assertIsNotNone(
                y_arr,
                msg="Dependent values array is None as the integration could not be performed."
            )

            # set up the validated x and y arrays using the known value of the integrated solution
            y_solution = initial_y * numpy.cos(w * x_arr) + initial_dy_dx * numpy.sin(w * x_arr) / w

            # set up variables to track convergence failures and the allowed error
            convergence_failures = False
            maximal_error = 1.0

            # iterate and find the maximal allowed error before encountering convergence failures
            while not convergence_failures:
                for i, __ in enumerate(x_arr):
                    # obtain the calculated value at the current x_value
                    y_calc = y_arr[0][i]

                    # obtain the exact value at the current x_value
                    y_known = y_solution[i]

                    # calculate the relative y_value to be compared with a value of 1
                    # Note: A relative y value of 1 indicates perfect convergence
                    relative_y = y_calc / y_known

                    # compare the relative y_value to perfect convergence
                    if abs(relative_y - 1) >= maximal_error:
                        convergence_failures = True
                        break

                # if not convergence failures are found, increase the precision
                if not convergence_failures:
                    maximal_error /= 10

            if not silent:
                # print a message including the maximal error for each integration technique
                print(
                    "Estimating the exact solution with {} using {} steps yielded no "
                    "convergence failures up to a maximal error of {}.".format(
                        integration_technique,
                        len(x_arr),
                        maximal_error
                    )
                )


NumericalIntegrationTestCase().test_integration_harm_oscill(silent=False)
