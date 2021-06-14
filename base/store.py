"""
Module to define methods for storing, generating, converting, and acquiring stored data.

:title: store.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

import json
import os
import pickle

# set the filepath separator value
seperator = os.path.sep

# set the filepath of the current file
curr_filepath = os.path.abspath(__file__)

# path of the default data store
default_data_dir = os.path.join(
    curr_filepath.replace(
        seperator + curr_filepath.split(seperator)[-2],
        ""
    ).replace(
        seperator + curr_filepath.split(seperator)[-1],
        ""
    ),
    "data"
)


class Store:
    """
    Class object defining methods used in data acquisition, storage, and maintenance.
    """

    def __init__(self, data_directory=default_data_dir):
        """
        Constructor class object for the Store class.

        :param data_directory: The path to the directory where data is stored
            (or is intended to be stored).
        """

        self.supported_data_formats = [".json", ".pickle", ".txt"]
        self.supported_plot_formats = [".png", ".pdf", ".svg", ".jpeg", ".jpg"]
        self.data_directory = data_directory

    def admin(self, verbose=False):
        """
        Method to print useful information about the data store.

        :param verbose: A boolean indicating if information about all the stored data
            is printed as well.
        """

        # print the administrative information
        print("Computational Cosmology Data Store")

        # acquire a list of the files and ancillary directories in the data store
        data_names = os.listdir(self.data_directory)

        # obtain a list of the file sizes of all data files in the data store
        data_sizes = []
        for name in data_names:
            filepath = os.path.join(self.data_directory, name)
            size = os.path.getsize(filepath)
            data_sizes.append(size)

        # print the number of files contained within the data store
        print("Files: {}".format(len(data_names)))

        # print the complete size of the data store
        print("Size: {} bytes".format(sum(data_sizes)))

        # if verbose, print all the available data files
        if verbose:
            # enumerate all the data files contained within the data store
            print("\nAll Data:")
            for i, name in enumerate(data_names):
                print("    {}: {} bytes".format(name, data_sizes[i]))

    def get(self, data_filename=""):
        """
        Method for acquiring data from the data store.

        :param data_filename: The name of the file containing the intended data.
        :returns: The data from the selected data file.
        """

        # construct the full filepath
        filepath = os.path.join(self.data_directory, data_filename)

        # extract the extension from the full filepath
        file_ext = os.path.splitext(filepath)[-1]

        # ensure that the intended data file exists and is of a supported file format
        if not all(
                [
                    os.path.exists(filepath), # the requested file exists
                    os.path.isfile(filepath), # the requested file is a file
                    file_ext in self.supported_data_formats # the requested file's type is supported
                ]
        ):
            raise IOError("Intended data file is not found or is incompatible.")

        # determine the type of file that is requested (ie. it's format)
        extension = os.path.splitext(filepath)[-1]

        if extension == ".pickle":
            # open the intended file and extract the data using the `pickle` package
            with open(filepath, 'rb') as open_file:
                data = pickle.load(open_file)
        elif extension == ".json":
            # open the intended file and extract the data using the built-in `json` package
            with open(filepath) as open_file:
                data = json.load(open_file)
        elif extension == ".txt":
            # open the intended file and extract the data using the built-in .read() method
            with open(filepath) as open_file:
                data = open_file.read()

        return data

    def save_data(self, data=None, data_filename=""):
        """
        Method for saving data locally. Permissions for overwriting data will be
        requested if necessary.

        :param data: The data intended to be saved.
        :param data_filename: The name of the file to contain the inputted data.
        """

        # acquire the full path to the intended data file
        filepath = os.path.join(self.data_directory, data_filename)

        # ensure the intended data file is of a supported file format
        extension = os.path.splitext(filepath)[-1]
        if extension not in self.supported_data_formats:
            print(
                "The provided filename indicates that the data is intended to be saved "
                "as an incompatible file format, '{}'. Instead, the data will be saved "
                "as a pickle file (with a `.pickle` extension).".format(extension)
            )
            # replace the filepath's old, incompatible extension with `.pickle`
            filepath = filepath.replace(extension, ".pickle")
            extension = ".pickle"

        # ensure the intended data file will not overwrite any existing files
        if os.path.exists(filepath):
            # include that the initial saving process was not done cleanly (possible overwrite)
            clean = False

            # in the event of an overwrite, ask the user if the data file should be overwritten
            print("The intended data file already exists in the selected directory.")
            overwrite = input("Overwrite? [Y], N >>> ").lower() in ["y", "yes", ""]
        else:
            # include that the initial saving process was performed cleanly
            clean = True

            # include that overwriting is allowed (as there is nothing being overwritten)
            overwrite = True

        if overwrite or clean:
            # ensure data is provided to be saved, otherwise nothing will happen
            if data is not None:
                # save the data using a method that is dependent on the data format
                if extension == ".pickle":
                    with open(filepath, 'wb') as open_file:
                        pickle.dump(data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
                elif extension == ".json":
                    with open(filepath, "w") as open_file:
                        json.dump(data, open_file)
                elif extension == ".txt":
                    with open(filepath, "w") as open_file:
                        open_file.write(data)
            else:
                print("No data was provided. Therefore, no data will be saved.")
        else:
            print("Overwrite Permission Denied. Data will not be saved")

    def save_plot(figure=None, plot_filename=""):
        """
        A method to save the provided figure in the provided filename.
        """

        # create the full filepath to the intended plot file
        filepath = os.path.join(self.data_directory, plot_filename)

        # ensure the provided plot filename is to be saved as a supported format
        extension = os.path.splitext(filepath)[-1]
        if extension not in self.supported_plot_formats:
            print(
                "The provided filename indicates that the data is intended to be saved "
                "as an incompatible file format, '{}'. Instead, the data will be saved "
                "as a png file (with a `.png` extension).".format(extension)
            )
            # replace the original filepath's extension with the default, `.png`
            filepath = filepath.replace(extension, ".png")
            extension = ".png"

        # ensure that saving the intended plot file will not overwrite any existing files
        if os.path.exists(filepath):
            # include that the initial saving process was not done cleanly (possible overwrite)
            clean = False

            # in the event of an overwrite, ask the user if the data file should be overwritten
            print("The intended data file already exists in the selected directory.")
            overwrite = input("Overwrite? [Y], N >>> ").lower() in ["y", "yes", ""]
        else:
            # include that the saving process was performed cleanly
            clean = True

            # include that overwriting is allowed (because no information is being overwritten)
            overwrite = True

        if overwrite or clean:
            # ensure there exists a figure to be saved
            if figure:
                # save that data using matplotlib.pyplot's savefig method
                figure.savefig(filepath)
            else:
                print("No figure was provided. Therefore, no plot will be saved.")
        else:
            print("Overwrite Permission Denied. Plot will not be saved.")
