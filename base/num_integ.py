"""
Module to define numerical integration techniques.

:title: num_integ.py

:author: Mitchell Shahen

:history: 02/05/2021

For simplicity, in all numerical integration techniques, the independent variable will be represented
by `x` and the dependent varaible will be represented by `y`. Additionally, the equation or function
that is to be integrated must be represented by a class object and contain a method called `function`.
Moreover, `function` must accept variables of `x` and `y` and return an array containing a number of
values. The inhomogeneous portion of the ODE must be calculated and available in the output array from
the `function` method. For second order ODEs, two values must be present in the output array, the
inhomogeneous portions of dy/dx and of dv/dx using dy/dx = v. It is this `function` that will be used
to calculate the solution.

An example of a valid class object, compatible with the `execution` method, for solving a first order
ODE is included below:

dy/dx = ((a ** 2) * x) + (b * y)

```py
class Sample:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def function(self, x, y):
        soln = ((self.a ** 2) * x) + (b * y)
        return array([soln])
```

An example of a valid class object, compatible with the `execution` method, for solving a second order
ODE is included below:

d2y/dx2 = ((a + x) * dy/dx) - (b / y)

Using dy/dx = v, we get two first order ODEs, one for y and one for v.

dy/dx = v
dv/dx = ((a + x) * v) - (b / y)

Note for second order ODEs, the y variable now has two components: y[0] = y, y[1] = v.

```py
class Sample2:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def function(self, x, y):
        soln_y = y[1]
        soln_v = ((self.a + x) * y[1]) - (self.b / y[0])
        return array([soln_y, soln_v])
```

One can adapt the above sample functions to solving any order of differential equation.

In addition to providing the function required to solve the differential equation, parameters
pertaining to the solution must also be included. These are the range of x values (`x_range`),
the step value (`h`), and the initial value(s) (`y_initial`).

Note: The x range must be a list of two values, the initial x value and the final x value in
    that order. Moreover, the final x value must be larger than the initial x value.

Note: The step value must be sufficiently small compared to the y value outputs.

Note: The initial value(s) must be a list containing initial values for each of the dependent
    variables being calculated. For example, in a second order ODE, y and dy/dx are being
    solved, therefore, two initial values (one for y and one for dy/dx) must be provided.
"""

import inspect
import numpy


class NumericalIntegration:
    """
    Class to define and execute numerical integration techniques for solving differential equations.
    """

    def __init__(self, technique=""):
        """
        Constructor class method for the Numerical Integration class.
        Includes supported integration techniques/methods.
        """

        # make all the supported techniques available
        self.supported_techniques = {
            "eulers_method": self.eulers_method,
            "heuns_method": self.heuns_method,
            "runge_kutta2": self.runge_kutta2,
            "runge_kutta4": self.runge_kutta4,
            "leapfrog_method": self.leapfrog_method
        }

        # make the selected technique available
        self.technique = technique

    @staticmethod
    def eulers_method(function=None, x_range=[], step_value=0, y_initial=[]):
        """
        Euler's method for solving initial value problems for numerical integration.
        In Euler's method, we employ a strategy similar to that of a Taylor series
        expansion, but neglecting the higher order, O(h**2), terms.
        """

        # set up the number of steps set to occur
        n_steps = int((x_range[1] - x_range[0]) / step_value)

        # set up an array to contain the independent variable
        x_arr = step_value * numpy.arange(n_steps)

        # set up another array to contain the dependent variable
        y_arr = numpy.empty([len(y_initial), len(x_arr)])

        # set up a buffer array to be updated with the results of the integration calculations
        y_buffer = numpy.array(y_initial)

        # perform the Euler integration
        for i, x_value in enumerate(x_arr):
            # write the buffer values to the dependent variable array
            y_arr[i] = y_buffer

            # calculate the results of the current step using the inputted function
            calc_results = function(x=x_value, y=y_buffer)

            # write the results of the calculation(s) to the buffer array
            y_buffer += calc_results

        return x_arr, y_arr

    @staticmethod
    def heuns_method(function=None, x_range=[], step_value=0, y_initial=[]):
        """
        Heun's method for numerical integration.
        """

        # set up the number of steps set to occur
        n_steps = int((x_range[1] - x_range[0]) / step_value)

        # set up an array to contain the independent variable
        x_arr = step_value * numpy.arange(n_steps)

        # set up another array to contain the dependent variable
        y_arr = numpy.empty([len(y_initial), len(x_arr)])

        # set up a buffer array to be updated with the results of the integration calculations
        y_buffer = numpy.array(y_initial)

        # perform the Euler integration
        for i, x_value in enumerate(x_arr):
            # write the buffer values to the dependent variable array
            y_arr[i] = y_buffer

            # calculate the next x_value needed for the calculation
            x_intermediate = x_value + step_value

            # calculate the intermediate y_buffer value
            y_intermediate = y_buffer + step_value * function(x=x_value, y=y_buffer)

            # calculate the results of the current step using the inputted function
            y_next = step_value * (
                    function(x=x_value, y=y_buffer) + function(x=x_intermediate, y=y_intermediate)
            ) / 2

            # write the results of the calculation(s) to the buffer array
            y_buffer += y_next

        return x_arr, y_arr

    @staticmethod
    def runge_kutta2(function=None, x_range=[], step_value=0, y_initial=[]):
        """
        2nd order Runge-Kutta method for numerical integration.
        """

        # set up the number of steps set to occur
        n_steps = int((x_range[1] - x_range[0]) / step_value)

        # set up an array to contain the independent variable
        x_arr = step_value * numpy.arange(n_steps)

        # set up another array to contain the dependent variable
        y_arr = numpy.empty([len(y_initial), len(x_arr)])

        # set up a buffer array to be updated with the results of the integration calculations
        y_buffer = numpy.array(y_initial)

        # perform the RK2 integration
        for i, x_value in enumerate(x_arr):
            # write the buffer values to the dependent variable array
            y_arr[i] = y_buffer

            # calculate the intermediate y_buffer value
            y_intermediate = step_value * function(x=x_value, y=y_buffer)

            # calculate the parameters needed to find the next y value
            x_param = x_value + 0.5 * step_value
            y_param = y_buffer + 0.5 * y_intermediate

            # calculate the intended next y value
            y_next = step_value * function(x=x_param, y=y_param)

            # write the results of the calculation(s) to the buffer array
            y_buffer += y_next

        return x_arr, y_arr

    @staticmethod
    def runge_kutta4(function=None, x_range=[], step_value=0, y_initial=[]):
        """
        4th order Runge-Kutta method for numerical integration.
        """

        # set up the number of steps set to occur
        n_steps = int((x_range[1] - x_range[0]) / step_value)

        # set up an array to contain the independent variable
        x_arr = step_value * numpy.arange(n_steps)

        # set up another array to contain the dependent variable
        y_arr = numpy.empty([len(y_initial), len(x_arr)])

        # set up a buffer array to be updated with the results of the integration calculations
        y_buffer = numpy.array(y_initial)

        # perform the RK4 integration
        for i, x_value in enumerate(x_arr):
            # write the buffer values to the dependent variable array
            y_arr[i] = y_buffer

            # calculate the four necessary intermediate y_buffer value
            y_intermediate1 = step_value * function(
                x=x_value,
                y=y_buffer
            )
            y_intermediate2 = step_value * function(
                x=x_value + 0.5 * step_value,
                y=y_buffer + 0.5 * y_intermediate1
            )
            y_intermediate3 = step_value * function(
                x=x_value + 0.5 * step_value,
                y=y_buffer + 0.5 * y_intermediate2
            )
            y_intermediate4 = step_value * function(
                x=x_value + step_value,
                y=y_buffer + y_intermediate3
            )

            # calculate the intended next y value
            y_next = (y_intermediate1 + 2 * y_intermediate2 + 2 * y_intermediate3 + y_intermediate4) / 6

            # write the results of the calculation(s) to the buffer array
            y_buffer += y_next

        return x_arr, y_arr

    @staticmethod
    def leapfrog_method(function=None, x_range=[], step_value=0, y_initial=[]):
        """
        Method to calculate the leapfrog method for numerical integration.
        """

        # set up the number of steps set to occur
        n_steps = int((x_range[1] - x_range[0]) / step_value)

        # set up an array to contain the independent variable
        x_arr = step_value * numpy.arange(n_steps)

        # set up another array to contain the dependent variable
        y_arr = numpy.empty([len(y_initial), len(x_arr)])

        # set up a buffer array to be updated with the results of the integration calculations
        y_buffer = numpy.array(y_initial)

        # set up an intermediary buffer array using 2nd order Runge-Kutta
        y_intermediary = y_buffer + 0.5 * step_value * function(
            x=x_range[0] + (step_value / 4),
            y=y_buffer + (step_value * function(x=x_range[0], y=y_buffer) / 4)
        )

        # perform the RK4 integration
        for i, x_value in enumerate(x_arr):
            # write the buffer values to the dependent variable array
            y_arr[i] = y_buffer

            # calculate the intended next y value and update the buffer array
            y_next = step_value * function(x=x_value + step_value / 2, y=y_intermediary)
            y_buffer += y_next

            # calculate the next intermediary value and update the intermediary buffer
            y_int_next = step_value * function(x=x_value + step_value, y=y_buffer)
            y_intermediary += y_int_next

        return x_arr, y_arr

    @staticmethod
    def _validate_parameters(technique="", supported=None, f_class=None, x_range=[], step_value=0, y_initial=[]):
        """
        Method to validate the inputted parameters intended to be executed.
        """

        # ensure the selected numerical integration technique is supported
        if technique not in supported.keys():
            raise IOError("Selected numerical integration technique not found.")

        # ensure that the f_class parameter is a class object and contains a method called "function"
        if not all(
                [
                    inspect.isclass(f_class),
                    "function" in dir(f_class)
                ]
        ):
            raise IOError("Invalid Function Class Provided.")

        # ensure that the x_range parameter is a list of two non-negative and increasing numbers
        if not all(
                [
                    isinstance(x_range, list),
                    len(x_range) == 2,
                    all([isinstance(x_value, (int, float)) for x_value in x_range])
                ]
        ):
            raise IOError("Inputted initial and final x value list is invalid/incompatible.")

        if not x_range[1] > x_range[0]:
            raise IOError("Input initial x value cannot be larger than the final x value.")

        # ensure the step value is a non-negative number
        if not isinstance(step_value, (int, float)):
            raise IOError("Inputted step value must be a number.")

        if not step_value > 0:
            raise IOError("Inputted step value cannot be negative.")

        # ensure that the y_initial is a list of at least one number
        if not all(
                [
                    isinstance(y_initial, list),
                    len(y_initial) > 0,
                    all([isinstance(initial_value, (int, float)) for initial_value in y_initial])
                ]
        ):
            raise IOError("Invalid initial y values list.")

    def execute(self, f_class=None, x_range=[], step_value=0, y_initial=[]):
        """
        Method to execute a selected numerical integration technique
        """

        # ensure all the necessary parameters are valid
        self._validate_parameters(
            technique=self.technique,
            supported=self.supported_techniques,
            f_class=f_class,
            x_range=x_range,
            step_value=step_value,
            y_initial=y_initial
        )

        # execute the intended integration method
        x_arr, y_arr = self.supported_techniques[self.technique](
            function=f_class.function,
            x_range=x_range,
            step_value=step_value,
            y_initial=y_initial
        )

        return x_arr, y_arr
