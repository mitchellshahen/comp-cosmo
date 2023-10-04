"""
Module to perform all the necessary functions and calculations to generate the datasets that
describe a star. Additionally, plots of the stellar structure variables are rendered.

:title: main.py

:author: Mitchell Shahen

:history: 10/05/2021
"""

from base.generate import generate_star, generate_star_sequence
from base.stellar_structure.base_stellar_structure import StellarStructureBase
from base.stellar_structure.grav_stellar_structure import StellarStructureGrav


# ---------- # SET THE AVAILABLE MODULES AND MODIFICATIONS # ---------- #

# create a dictionary of all the supported functions
AVAIL_FUNCTIONS = {
    "Generate a Star": generate_star,
    "Generate a Stellar Sequence": generate_star_sequence
}

# create a dictionary of all the available stellar structure modifications
AVAIL_MODS = {
    "No Modifications": StellarStructureBase,
    "Gravity Modifications": StellarStructureGrav
}

def execute():
    """
    Function to execute a method of analysis as specified by the user.
    """

    # ---------- # ASK THE USER TO SELECT A FUNCTION AND STELLAR STRUCTURE # ---------- #

    # print the available functions that the user may select
    print("\nAll Available Functions:")
    for i, func in enumerate(AVAIL_FUNCTIONS):
        print(f"[{i+1}]: {func}")

    # instruct the user to select a function from those listed
    select_func_num = input("\nSelect a Function Number (Press [Enter] to Exit) >>> ")
    if len(select_func_num) == 0 or select_func_num.lower() == "q":
        print("Aborting program...")
        exit()
    else:
        select_func = list(AVAIL_FUNCTIONS)[int(select_func_num) - 1]
        select_func_exec = AVAIL_FUNCTIONS[select_func]

    # print the available modifications that the user may select
    print("\nAll Available Stellar Structure MOdifications:")
    for i, mod in enumerate(AVAIL_MODS):
        print(f"[{i+1}]: {mod}")

    # instruct the user to select a function from those listed
    select_mod_num = input("\nSelect a Modification Number (Press [Enter] to Exit) >>> ")
    if len(select_mod_num) == 0 or select_func_num.lower() == "q":
        print("Aborting program...")
        exit()
    else:
        select_mod = list(AVAIL_MODS)[int(select_mod_num) - 1]
        select_mod_exec = AVAIL_MODS[select_mod]

    # execute the selected function using the selected stellar structure class
    print(f"\nExecuting '{select_func}' Function with '{select_mod}'...")
    select_func_exec(stellar_structure=select_mod_exec)


if __name__ == "__main__":
    execute()
