"""
Module to set up unittesting of the Numerical Integration module: `base.num_integ.NumericalIntegration`
"""

import numpy
import sys
import unittest

# access the comp-cosmo directory
sys.path.append("../../../")

from base.num_integ import NumericalIntegration

# ---------- # VALID NUMERICALINTEGRATION PARAMETERS # ---------- #

# valid numerical integration technique
valid_technique = "eulers_method"


# supported Function Class
class ValidFClass:
    def __init__(self, a):
        self.a = a

    def function(self, x, y):
        return numpy.array((-1.0 * self.a * y))


# valid range of independent variable values
valid_x_range = [0.0, 1.0]

# valid non-zero step value
valid_step_value = 1.0

# valid initial dependent variable value(s)
valid_y_initial = [0.0]


class NumericalIntegrationTestCase(unittest.TestCase):
    """
    Unit testing class for the NumericalIntegration class.
    """

    def test_validate_parameters(self):
        """
        Method to test the validate parameters functionality.
        """

        # test that all valid parameters will cause the validate function to allow proceeding
        proceed_all_valid = NumericalIntegration()._validate_parameters(
            technique=valid_technique,
            f_class=ValidFClass(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertTrue(proceed_all_valid)

        # test that the validate function will notice an erroneous integration technique
        unsupported_technique = "nothing"
        proceed_bad_technique = NumericalIntegration()._validate_parameters(
            technique=unsupported_technique,
            f_class=ValidFClass(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_technique)

        # test that the validate function will notice an erroneous function class
        class InvalidFClass:
            pass

        proceed_bad_f_class = NumericalIntegration()._validate_parameters(
            technique=valid_technique,
            f_class=InvalidFClass(),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_f_class)

        # test that the validate function will notice an erroneous x_range list
        invalid_x_range = []
        proceed_bad_x_range = NumericalIntegration()._validate_parameters(
            technique=valid_technique,
            f_class=ValidFClass(a=0.0),
            x_range=invalid_x_range,
            step_value=valid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_x_range)

        # test that the validate function will notice an erroneous step value input
        invalid_step_value = 0
        proceed_bad_step_value = NumericalIntegration()._validate_parameters(
            technique=valid_technique,
            f_class=ValidFClass(a=0.0),
            x_range=valid_x_range,
            step_value=invalid_step_value,
            y_initial=valid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_step_value)

        # test that the validate function will notice an erroneous initial y_value list
        invalid_y_initial = ["hello", "world"]
        proceed_bad_y_initial = NumericalIntegration()._validate_parameters(
            technique=valid_technique,
            f_class=ValidFClass(a=0.0),
            x_range=valid_x_range,
            step_value=valid_step_value,
            y_initial=invalid_y_initial,
            silent=True
        )
        self.assertFalse(proceed_bad_y_initial)

    def test_eulers_method(self):
        """
        Unit test to test that Euler's Method integrates and approximates an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = 10.0

        # integrate the differential equation using Euler's Method
        x_arr, y_arr = NumericalIntegration(technique="eulers_method").execute(
            f_class=ValidFClass(a=differential_coeff),
            x_range=[0.0, 10.0],
            step_value=0.001,
            y_initial=[initial_value],
            silent=True
        )

        # ensure no validation errors have occurred
        self.assertIsNotNone(x_arr)
        self.assertIsNotNone(y_arr)

        # set up the validated x and y arrays using the known value of the integrated solution
        y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

        # iterate and find the maximal allowed error before encountering convergence failures
        convergence_failures = False
        maximal_error = 1.0
        while not convergence_failures:
            for i, y_calc in enumerate(y_arr[0]):
                y_known = y_solution[i]
                if abs(y_calc - y_known) >= maximal_error:
                    convergence_failures = True
                    break
            if not convergence_failures:
                maximal_error /= 10

        print(
            "Estimating the exact solution with Euler's Method using {} steps yielded no "
            "convergence failures up to a maximal error of {}.".format(len(x_arr), maximal_error)
        )

    def test_heuns_method(self):
        """
        Unit test to test that Heun's Method integrates and approximates an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = 10.0

        x_arr, y_arr = NumericalIntegration(technique="heuns_method").execute(
            f_class=ValidFClass(a=differential_coeff),
            x_range=[0.0, 10.0],
            step_value=0.001,
            y_initial=[initial_value],
            silent=True
        )
        self.assertIsNotNone(x_arr)
        self.assertIsNotNone(y_arr)

        # set up the validated x and y arrays using the known value of the integrated solution
        y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

        # iterate and find the maximal allowed error before encountering convergence failures
        convergence_failures = False
        maximal_error = 1.0
        while not convergence_failures:
            for i, y_calc in enumerate(y_arr[0]):
                y_known = y_solution[i]
                if abs(y_calc - y_known) >= maximal_error:
                    convergence_failures = True
                    break
            if not convergence_failures:
                maximal_error /= 10

        print(
            "Estimating the exact solution with Heun's Method using {} steps yielded no "
            "convergence failures up to a maximal error of {}.".format(len(x_arr), maximal_error)
        )

    def test_runge_kutta2(self):
        """
        Unit test to test that 2nd-Order Runge-Kutta integrates and approximates an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = 10.0

        x_arr, y_arr = NumericalIntegration(technique="runge_kutta2").execute(
            f_class=ValidFClass(a=differential_coeff),
            x_range=[0.0, 10.0],
            step_value=0.001,
            y_initial=[initial_value],
            silent=True
        )
        self.assertIsNotNone(x_arr)
        self.assertIsNotNone(y_arr)

        # set up the validated x and y arrays using the known value of the integrated solution
        y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

        # iterate and find the maximal allowed error before encountering convergence failures
        convergence_failures = False
        maximal_error = 1.0
        while not convergence_failures:
            for i, y_calc in enumerate(y_arr[0]):
                y_known = y_solution[i]
                if abs(y_calc - y_known) >= maximal_error:
                    convergence_failures = True
                    break
            if not convergence_failures:
                maximal_error /= 10

        print(
            "Estimating the exact solution with 2nd-Order Runge-Kutta using {} steps yielded no "
            "convergence failures up to a maximal error of {}.".format(len(x_arr), maximal_error)
        )

    def test_runge_kutta4(self):
        """
        Unit test to test that 4th-Order Runge-Kutta integrates and approximates an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = 10.0

        x_arr, y_arr = NumericalIntegration(technique="runge_kutta4").execute(
            f_class=ValidFClass(a=differential_coeff),
            x_range=[0.0, 10.0],
            step_value=0.001,
            y_initial=[initial_value],
            silent=True
        )
        self.assertIsNotNone(x_arr)
        self.assertIsNotNone(y_arr)

        # set up the validated x and y arrays using the known value of the integrated solution
        y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

        # iterate and find the maximal allowed error before encountering convergence failures
        convergence_failures = False
        maximal_error = 1.0
        while not convergence_failures:
            for i, y_calc in enumerate(y_arr[0]):
                y_known = y_solution[i]
                if abs(y_calc - y_known) >= maximal_error:
                    convergence_failures = True
                    break
            if not convergence_failures:
                maximal_error /= 10

        print(
            "Estimating the exact solution with 4th-Order Runge-Kutta using {} steps yielded no "
            "convergence failures up to a maximal error of {}.".format(len(x_arr), maximal_error)
        )

    def test_leapfrog_method(self):
        """
        Unit test to test that the Leapfrog Method integrates and approximates an exponential function:
            df/dx = -1.0 * a * f(x)
        The above differential equation yields the following when integrated:
            f(x) = f(0) * exp(-1.0 * a * x)
        We will set f(0) = 10, a = 0.5, and integrate from x = 0 to x = 10.
        """

        # set the constants required to evaluate this method
        differential_coeff = 0.5
        initial_value = 10.0

        x_arr, y_arr = NumericalIntegration(technique="leapfrog_method").execute(
            f_class=ValidFClass(a=differential_coeff),
            x_range=[0.0, 10.0],
            step_value=0.001,
            y_initial=[initial_value],
            silent=True
        )
        self.assertIsNotNone(x_arr)
        self.assertIsNotNone(y_arr)

        # set up the validated x and y arrays using the known value of the integrated solution
        y_solution = initial_value * numpy.exp(-1.0 * differential_coeff * x_arr)

        # iterate and find the maximal allowed error before encountering convergence failures
        convergence_failures = False
        maximal_error = 1.0
        while not convergence_failures:
            for i, y_calc in enumerate(y_arr[0]):
                y_known = y_solution[i]
                if abs(y_calc - y_known) >= maximal_error:
                    convergence_failures = True
                    break
            if not convergence_failures:
                maximal_error /= 10

        print(
            "Estimating the exact solution with the Leapfrog Method using {} steps yielded no "
            "convergence failures up to a maximal error of {}.".format(len(x_arr), maximal_error)
        )


# NumericalIntegrationTestCase().test_validate_parameters()
NumericalIntegrationTestCase().test_eulers_method()
NumericalIntegrationTestCase().test_heuns_method()
NumericalIntegrationTestCase().test_runge_kutta2()
NumericalIntegrationTestCase().test_runge_kutta4()
NumericalIntegrationTestCase().test_leapfrog_method()
