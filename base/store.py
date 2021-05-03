"""
Module to define methods for storing, generating, converting, and acquiring stored data.

:title: store.py

:author: Mitchell Shahen

:history: 02/05/2021
"""

import os
import pickle

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
    Class object defining methods used in the acquisition, storage, and maintenance of available data.
    """

    def __init__(self, data_directory=default_data_dir):
        """
        Constructor class object for the Store class.
        """

        self.data_directory = data_directory

    def acquire(self, data_filename=""):
        """
        Method for acquiring data
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
        Method for saving data locally
        """

        filepath = os.path.join(self.data_directory, data_filename)

        # ensure the intended data file is a pickle file
        if os.path.splitext(filepath)[-1] != ".pickle":
            print("Intended data file will be saved as a pickle file.")
            filepath = filepath.replace(os.path.splitext(filepath)[-1], ".pickle")

        # ensure the intended data file will not overwrite any existing files
        increment = 0
        basename = os.path.splitext(filepath.split("\\")[-1])[0]
        while os.path.exists(filepath):
            filepath = filepath.replace(
                os.path.splitext(filepath.split("\\")[-1])[0],
                "{}_{}".format(basename, increment)
            )
            increment += 1

        # open the intended data file and save the data
        with open(filepath, 'wb') as open_file:
            pickle.dump(data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
