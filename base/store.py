"""
Module to define methods for storing, generating, converting, and acquiring stored data.

:title: store.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

import os
import pickle

# path of the default data store
default_data_dir = os.path.join(
    __file__.replace(
        "\\" + __file__.split("\\")[-2],
        ""
    ).replace(
        "\\" + __file__.split("\\")[-1],
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

        # ensure the intended data file exists and is a pickle file
        if not all(
                [
                    os.path.exists(filepath),
                    os.path.isfile(filepath),
                    filepath.split(".")[-1] == "pickle"
                ]
        ):
            raise IOError("Intended data file is not found or is incompatible.")

        # open the intended file and extract the data
        with open(filepath, 'rb') as open_file:
            data = pickle.load(open_file)

        return data

    def save(self, data=None, data_filename=""):
        """
        Method for saving data locally.

        :param data: The data intended to be saved.
        :param data_filename: The name of the file to contain the inputted data.
        """

        filepath = os.path.join(self.data_directory, data_filename)

        # ensure the intended data file is a pickle file
        if os.path.splitext(filepath)[-1] != ".pickle":
            print("Intended data file will be saved as a pickle file.")
            filepath = filepath.replace(os.path.splitext(filepath)[-1], ".pickle")

        # ensure the intended data file will not overwrite any existing files
        basename = os.path.splitext(filepath.split("\\")[-1])[0]
        if os.path.exists(filepath):
            # in the event of an overwrite, ask if the data file should be overwritten
            print(
                "The intended data file, {}, already exists "
                "in the selected directory.".format(basename)
            )
            clean = False
            overwrite = input("Overwrite? [Y], N >>> ").lower() in ["y", "yes", ""]
        else:
            clean = True
            overwrite = True

        if overwrite or clean:
            # open the intended data file and save the data
            with open(filepath, 'wb') as open_file:
                pickle.dump(data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
