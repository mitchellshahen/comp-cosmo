"""
Plotting Module

:title: plot.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

import matplotlib.pyplot as plt
import numpy


class Plotting:
    """
    Class object for plotting inputted data.
    """

    def __init__(self, plot_type=""):
        """
        Constructor class object for the Plotting class
        """

        # include all the valid types of plots
        self.supported_plot_types = {
            "line": self._line_plot,
            "scatter": self._scatter_plot
        }

        # acquire the selected plot type
        self.plot_type = plot_type

    @staticmethod
    def _line_plot(data=None, params={}):
        """
        Method to define a line plot that is to be rendered. The line plot is intended to recieve an
        x-component dataset and a y-component dataset, each of which is a list or a numpy ndarray.
        Therefore, the `data` parameter is either a nested list of two items or a 2D numpy array.

        :param data: A 2-Dimensional numpy ndarray or nested list containing data to be plotted.
        :param params: A dictionary specifying parameters used in plotting or rendering the data.
        """

        # set all of the available plotting parameters for the line plot
        available_params = ["title", "xlabel", "ylabel"]

        # enusre the data input is of the proper form
        if isinstance(data, numpy.ndarray):
            if data.shape[0] != 2:
                raise IOError(
                    "Data array is {}-dimensional when it must be 2-dimensional.".format(
                        data.shape[0]
                    )
                )
            elif data.shape[1] == 0:
                raise IOError("The inputted data array contains no elements.")
        else:
            if len(data) != 2:
                raise IOError(
                    "Data array is {}-dimensional when it must be 2-dimensional".format(
                        len(data)
                    )
                )

        # acquire the intended x-component and y-component for the scatter plot
        x_data = data[0]
        y_data = data[1]

        # ensure that all the inputted parameters are valid, invalid parameters are deleted
        for parameter in list(params.keys()):
            if parameter not in available_params:
                __ = params.pop(parameter)

        # fill out the plotting parameters dictionary with the available plotting parameters
        for parameter in available_params:
            if parameter not in list(params.keys()):
                params[parameter] = None

        # plot and render the data
        plt.plot(x_data, y_data)
        plt.title(params["title"])
        plt.xlabel(params["xlabel"])
        plt.ylabel(params["ylabel"])
        plt.show()

    @staticmethod
    def _scatter_plot(data=None, params={}):
        """
        Method to define a scatter plot to be rendered. The scatter plot is intended to recieve an
        x-component dataset and a y-component dataset, each of which is a list or a numpy ndarray.
        Therefore, the `data` parameter is either a nested list of two items or a 2D numpy array.

        :param data: A 2-Dimensional numpy ndarray or nested list containing data to be plotted.
        :param params: A dictionary specifying parameters used in plotting or rendering the data.
        """

        # set all of the available plotting parameters for the scatter plot
        available_params = ["title", "xlabel", "ylabel"]

        # enusre the data input is of the proper form
        if isinstance(data, numpy.ndarray):
            if data.shape[0] != 2:
                raise IOError(
                    "Data array is {}-dimensional when it must be 2-dimensional.".format(
                        data.shape[0]
                    )
                )
            elif data.shape[1] == 0:
                raise IOError("Data array contains no elements.")
        else:
            if len(data) != 2:
                raise IOError(
                    "Data array is {}-dimensional when it must be 2-dimensional".format(
                        len(data)
                    )
                )

        # acquire the intended x-component and y-component for the scatter plot
        x_data = data[0]
        y_data = data[1]

        # ensure that all the inputted parameters are valid, invalid parameters are deleted
        for parameter in list(params.keys()):
            if parameter not in available_params:
                __ = params.pop(parameter)

        # fill out the plotting parameters dictionary with the available plotting parameters
        for parameter in available_params:
            if parameter not in list(params.keys()):
                params[parameter] = None

        # plot and render the data
        plt.scatter(x=x_data, y=y_data)
        plt.title(params["title"])
        plt.xlabel(params["xlabel"])
        plt.ylabel(params["ylabel"])
        plt.show()

    def _validate(self, in_data=None, params={}):
        """
        Method to validate the parameters inputted to the execute function.

        :param in_data: An n-dimensional numpy ndarray or nested list containing data to be plotted.
        :param params: A dictionary specifying parameters used in plotting or rendering the data.
        :return: A boolean indicating if plotting can proceed with the given parameters.
        """

        # set up a boolean to track if plotting can proceed
        proceed = True

        # ensure the selected plot type is supported
        if self.plot_type not in self.supported_plot_types.keys():
            proceed = False
            print("Plot Type '{}' is not supported".format(self.plot_type))

        # ensure the inputted dataset is valid and contains plot-able elements
        if in_data is None:
            proceed = False
            print("ERROR: An input dataset has not been provided.")
        elif isinstance(in_data, numpy.ndarray):
            if in_data.shape[0] == 0:
                proceed = False
                print("ERROR: The inputted dataset must not be 0-Dimensional.")
        elif isinstance(in_data, list):
            if len(in_data) == 0:
                proceed = False
                print("ERROR: The inputted dataset must not be 0-Dimensional.")
        else:
            proceed = False
            print("ERROR: The inputted dataset must be a numpy.ndarray or a nested list.")

        # ensure the params parameter is a dict type
        if not isinstance(params, dict):
            proceed = False
            print("Invalid Parameters Type. Must be a dict object of plotting parameters.")

        return proceed

    def plot(self, in_data=None, params={}):
        """
        Method to execute the specified plot type with a given dataset and plotting parameters.

        :param in_data: An n-dimensional numpy ndarray or nested list containing data to be plotted.
        :param params: A dictionary specifying parameters used in plotting or rendering the data.
        """

        # validate all the inputted parameters
        proceed = self._validate(in_data=in_data, params=params)

        if proceed:
            # plot and render the selected plot type with the inputted validated dataset(s)
            self.supported_plot_types[self.plot_type](data=in_data, params=params)
        else:
            print("Parameter Test Failed. Unable to Plot the Given Dataset.")
