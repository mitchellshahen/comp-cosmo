"""
Plotting Module

:title: plot.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

from constants import L_sun
import matplotlib.pyplot as plt
import numpy

# ---------- # GENERAL PURPOSE PLOTTING CLASS # ---------- #


class Plotting:
    """
    Class object for plotting inputted data.
    """

    def __init__(self, plot_type="line"):
        """
        Constructor class object for the Plotting class

        :param plot_type: A string indicating the type of plot to create. Must be supported.
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


# ---------- # SPECIFIC PLOTTING FUNCTIONS # ---------- #

def stellar_structure_plot(radius, density, temperature, mass, luminosity):
    """
    Function to plot normalized stellar density, temperature, mass, and luminosity against radius.

    :param radius: An array of radius values.
    :param density: An array of density values.
    :param temperature: An array of temperature values.
    :param mass: An array of mass values.
    :param luminosity: An array of luminosity values.
    """

    # plot the stellar density against stellar radius
    plt.plot(radius, density, label="Density", color="black", linestyle="solid")

    # plot the stellar temperature against stellar radius
    plt.plot(radius, temperature, label="Temperature", color="red", linestyle="dashed")

    # plot the stellar mass against stellar radius
    plt.plot(radius, mass, label="Mass", color="green", linestyle="dashed")

    # plot the stellar luminosity against stellar radius
    plt.plot(radius, luminosity, label="Luminosity", color="blue", linestyle="dotted")

    # set the title
    plt.title("Stellar Structure Plot")

    # set the legend
    plt.legend()

    # render the plot
    plt.show()


def hr_diagram(effect_temps, luminosities):
    """
    Function to plot a Hertzsprung-Russell diagram, that is a plot of effective surface temperature
    against relative luminosity. The effective surface temperature is in Kelvin and the luminosity
    is relative to the Sun's luminosity. The luminosity is also commonly given as a logarithm.

    :param effect_temps: An array of effective surface temperatures for many stars.
    :param luminosities: An array of non-normalized luminosities for many stars.
    """

    # normalize the luminosities with the Sun's luminosity
    for i, star_luminosity in enumerate(luminosities):
        norm_lumin = star_luminosity / L_sun
        luminosities[i] = norm_lumin

    # plot the effective surface temperatures against the stellar luminosities
    plt.plot(effect_temps, luminosities, color="black", linestyle="solid")

    # set the labels for the x-axis and y-axis
    plt.xlabel("Effective Surface Temperature (in Kelvin)")
    plt.ylabel("Relative Luminosity, L/L_sun")

    # invert the x-axis so it is decreasing in temperature going left to right
    plt.gca().invert_xaxis()

    # set the y-axis (luminosities) to be logarithmic
    plt.yscale("log")

    # set the title
    plt.title("Hertzsprung-Russell Diagram")

    # render the plot
    plt.show()
