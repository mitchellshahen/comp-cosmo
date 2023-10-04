"""
Module to define the stellar structure equations that govern stellar bodies.

:title: base_stellar_structure.py

:author: Mitchell Shahen

:history: 05/05/2021
"""

import numpy

from base.constants import a, c, G, gamma, h_bar, k, m_e, m_p
from base.units import kg, W, m
from base.solve_stellar import RHO_INDEX, T_INDEX, M_INDEX, L_INDEX

# set the mass fractions of Hydrogen, Helium, and then the other elements
X_FRAC = 0.7381
Y_FRAC = 0.2485
Z_FRAC = 0.0134

# set up variables for introducing gravity modifications
# Note: the default gravity modifications are no modifications (grav_small = 0 and grav_large = inf)
GRAV_SMALL = 0
GRAV_LARGE = numpy.inf


class StellarStructureGrav:
    """
    Class object to define and solve the stellar structure equations.
    """

    def __init__(self):
        """
        Constructor class object for the StellarStructure class.
        Sets the mass fractions and gravitational effect parameters
        """

        # ensure that the user defined variables are valid and make sense
        if not all(
                [
                    X_FRAC + Y_FRAC + Z_FRAC >= 0.9999,
                    X_FRAC + Y_FRAC + Z_FRAC <= 1.0001,
                    GRAV_SMALL < GRAV_LARGE,
                    GRAV_SMALL >= 0,
                    GRAV_LARGE >= 0
                ]
        ):
            raise IOError("The inputted mass fractions and gravitational effects are invalid.")

        # set the mass fraction of hydrogen, helium, and all other remaining elements
        self.X = X_FRAC
        self.Y = Y_FRAC
        self.Z = Z_FRAC

        # set the lambda constraints
        self.lambda_small = GRAV_SMALL
        self.lambda_large = GRAV_LARGE

    def initial_properties(self, r_0, rho_0, T_0):
        """
        Method to define the initial values used to solve the stellar structure equations.

        :param r_0: The initial radius value used in solving the stellar structure equations.
        :param rho_0: The central density of the star.
        :param T_0: The central temperature of the star.
        :returns: A numpy ndarray of the initial stellar properties: central density, central
            temperature, initial mass, initial radiative luminosity, and initial optical density.
        """

        # set the mass contained within a sphere of very small radius
        M_0 = 4 * numpy.pi * (r_0 ** 3) * rho_0 / 3

        # set the luminosity generated by a sphere of very small radius
        L_0 = 4 * numpy.pi * (r_0 ** 3) * rho_0 * self.energy_gen_rate(rho=rho_0, T=T_0) / 3

        # set the initial optical density
        tau_0 = self.mean_opacity(rho=rho_0, T=T_0) * rho_0

        # amalgamate all the state variables into a single array
        initial_state = numpy.array([rho_0, T_0, M_0, L_0, tau_0])

        return initial_state

    def get_derivative_state(self, r, state):
        """
        Method to set up and acquire the derivatives of the state variables: density, temperature,
        mass, luminosity, and optical depth at a given radius.

        :param r: A radius at which the derivatives of various stellar properties are required.
        :param state: A matrix containing 5 arrays: the density array, temperature array, mass
            array, luminosity array, and optical depth array.
        :return: A matrix of 5 arrays containing the radial derivatives of density, temperature,
            mass, luminosity, and optical depth, respectively, all evaluated at `r`.
        """

        # extract from the input state the necessary variables
        rho = state[RHO_INDEX]
        T = state[T_INDEX]
        M = state[M_INDEX]
        L = state[L_INDEX]

        # ensure no state values are 0
        if any(
            [rho <= 0, T <= 0, M <= 0, L <= 0]
        ):
            raise ArithmeticError(
                "ERROR: Encountered a zero or negative state variable in the process of "
                "integrating the stellar structure equations."
            )

        # get the density derivative
        drho_dr = self.hydrostat_equil(r=r, rho=rho, T=T, M=M, L=L)

        # get the temperature derivative
        dT_dr = self.temperature_grad(r=r, rho=rho, T=T, M=M, L=L)

        # get the mass derivative
        dM_dr = self.mass_continuity(r=r, rho=rho)

        # get the luminosity derivative
        dL_dr = self.energy_equation(r=r, rho=rho, T=T, energy_gen=None)

        # get the optical depth derivative
        dtau_dr = self.optical_depth_equation(rho=rho, T=T)

        # set up the state variable
        deriv_state = numpy.array([drho_dr, dT_dr, dM_dr, dL_dr, dtau_dr])

        return deriv_state

    # ---------- # ANCILLARY CONSTANTS # ---------- #

    def pp_chain_energy(self, rho, T):
        """
        Method to calculate the energy generation rate exclusively from the PP-chain.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :return: The energy generation rate from the PP-chain.
        """

        # set the scaled density value
        rho_5 = rho / (10 ** 5)

        # set the scaled temperature value
        T_6 = T / (10 ** 6)

        # calculate the energy generation rate from the PP-chain
        epsilon_pp = (1.07e-7 * (W / kg)) * rho_5 * (self.X ** 2) * (T_6 ** 4)

        return epsilon_pp

    def cno_cycle_energy(self, rho, T, X_cno=None):
        """
        Method to calculate the energy generation rate exclusively from the CNO cycle.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param X_cno: The mass fraction of hydrogen due to the CNO cycle.
        :return: The energy generation rate from the CNO cycle.
        """

        # set the CNO abundance value
        if any(
                [
                    X_cno is None,
                    not isinstance(X_cno, (int, float))
                ]
        ):
            # if the included CNO abundance is invalid, use the solar CNO abundance
            X_cno = 0.03 * self.X

        # set the scaled density value
        rho_5 = rho / (10 ** 5)

        # set the scaled temperature value
        T_6 = T / (10 ** 6)

        # calculate the energy generate rate from the CNO cycle
        epsilon_cno = (8.24e-26 * (W / kg)) * rho_5 * self.X * X_cno * (T_6 ** 19.9)

        return epsilon_cno

    def energy_gen_rate(self, rho, T, X_cno=None):
        """
        Method to calculate the total energy generation rate from the PP-chain and the CNO cycle.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param X_cno: The mass fraction of hydrogen due to the CNO cycle.
        :return: The total energy generation rate.
        """

        epsilon_pp = self.pp_chain_energy(rho=rho, T=T)

        # calculate the energy generate rate from the CNO cycle
        epsilon_cno = self.cno_cycle_energy(rho=rho, T=T, X_cno=X_cno)

        # calculate the total energy generate rate
        epsilon_total = epsilon_pp + epsilon_cno

        return epsilon_total

    def mean_molec_weight(self):
        """
        Method to calculate the mean molecular weight as a function of the mass fractions.

        :returns: The mean molecular weight.
        """

        mu_total = (2 * self.X + 0.75 * self.Y + 0.5 * self.Z) ** (-1)

        return mu_total

    def opacity_e_scatter(self):
        """
        Method to calculate the opacity from electron scattering.

        :return: The electron scattering opacity.
        """

        # calculate the mean opacity from electron scattering
        kappa_es_coeff = 0.02 * (m ** 2 / kg)
        kappa_es = kappa_es_coeff * (1 + self.X)

        return kappa_es

    def opacity_ff_scatter(self, rho, T):
        """
        Method to calculate the opacity from free-free scattering.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :return: The free-free scattering opacity.
        """

        # set the scaled density value
        rho_3 = rho / (10 ** 3)

        # calculate the mean opacity from free-free scattering
        kappa_ff_coeff = 1.0e24 * (m ** 2 / kg)
        kappa_ff = kappa_ff_coeff * (self.Z + 0.0001) * (rho_3 ** 0.7) * (T ** (-7 / 2))

        return kappa_ff

    def opacity_hm_scatter(self, rho, T):
        """
        Method to calculate the opacity from H-minus scattering.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :return: The H-minus scattering opacity.
        """

        # set the scaled density value
        rho_3 = rho / (10 ** 3)

        # calculate the mean opacity from H-minus scattering
        kappa_hm_coeff = 2.5e-32 * (m ** 2 / kg)
        kappa_hm = kappa_hm_coeff * (self.Z / 0.02) * (rho_3 ** 0.5) * (T ** 9)

        return kappa_hm

    def mean_opacity(self, rho, T):
        """
        Method to calculate the Rossland mean opacities from three dominant processes:
            - Electron scattering
            - Free-free scattering (a Kramer-like approximation)
            - H-minus scattering at low temperatures

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :returns: The Rossland mean opacity.
        """

        # calculate the mean opacity from electron scattering
        kappa_es = self.opacity_e_scatter()

        # calculate the mean opacity from free-free scattering
        kappa_ff = self.opacity_ff_scatter(rho=rho, T=T)

        # calculate the mean opacity from H-minus scattering
        kappa_hm = self.opacity_hm_scatter(rho=rho, T=T)

        # calculate the total mean opacity
        kappa_total = ((1 / kappa_hm) + (1 / max(kappa_es, kappa_ff))) ** (-1)

        return kappa_total

    # ---------- # THERMODYNAMIC VARIABLES # ---------- #

    @staticmethod
    def degeneracy_pressure(rho):
        """
        Method to calculate the pressure of each relevant equation of state using a given density
        and temperature.

        :param rho: The stellar density.
        :returns: The degeneracy pressure in a stellar body.
        """

        # calculate the non-relativistic degeneracy pressure
        pressure_deg = (
            (3 * numpy.pi ** 2) ** (2/3)
        ) * (
            h_bar ** 2
        ) * (
            (rho / m_p) ** (5/3)
        ) / (
            5 * m_e
        )

        return pressure_deg

    @staticmethod
    def gas_pressure(rho, T, mu=1.0):
        """
        Method to calculate the pressure of each relevant equation of state using a given density
        and temperature.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param mu: The mean molecular weight.
        :returns: The pressure in a stellar body due to gaseous contributions.
        """

        # calculate the pressure generated by ideal gases
        pressure_gas = rho * k * T / (mu * m_p)

        return pressure_gas

    @staticmethod
    def photon_pressure(T):
        """
        Method to calculate the pressure of each relevant equation of state using a given density
        and temperature.

        :param T: The stellar temperature.
        :returns: The pressure in a stellar body due to photon gas contributions.
        """

        # calculate the photon gas pressure
        pressure_photon = a * (T ** 4) / 3

        return pressure_photon

    def total_pressure(self, rho, T):
        """
        Method to calculate the pressure of each relevant equation of state using a given density
        and temperature.

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :returns: The total pressure in a stellar body.
        """

        # calculate the non-relativistic degeneracy pressure
        pressure_deg = self.degeneracy_pressure(rho=rho)

        # calculate the pressure generated by ideal gases
        pressure_gas = self.gas_pressure(rho=rho, T=T, mu=self.mean_molec_weight())

        # calculate the photon gas pressure
        pressure_photon = self.photon_pressure(T=T)

        # calculate the total pressure as the sum of all the pressure components
        total_pressure = pressure_deg + pressure_gas + pressure_photon

        return total_pressure

    def temp_convective(self, r, rho, T, M):
        """
        Method to calculate the temperature due to convective contributions:
            convective temperature = (1 - 1/lambda) * T * G * M * rho / (P * r ** 2)

        :param r: The radius value for which the density, temperature, and enclosed mass correspond.
        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param M: The cumulative mass contained within a radius, `r`.
        :returns: The temperature due to convective contributions.
        """

        # get the total pressure
        pressure = self.total_pressure(rho=rho, T=T)

        # calculate the gravitational component
        grav_comp = G * M * (1 + (self.lambda_small / r) + (r / self.lambda_large)) / (r ** 2)

        # calculate the convective temperature gradient
        T_convective = (1 - 1 / gamma) * T * grav_comp * rho / pressure

        return T_convective

    def temp_radiative(self, r, rho, T, L):
        """
        Method to calculate the temperature due to radiative contributions:
            radiative temperature = 3 * kappa * rho * L / (16 * pi * a * c * T ** 3 * r ** 2)

        :param r: The radius value for which the density, temperature, and cumulative
            luminosity correspond.
        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param L: The cumulative luminosity radiated by the star, but considering contributions
            from inside a radius, `r`.
        :returns: The temperature due to radiative contributions.
        """

        # get the total mean opacity
        opacity = self.mean_opacity(rho=rho, T=T)

        # calculate the radiative temperature
        T_radiative = 3 * opacity * rho * L / (16 * numpy.pi * a * c * (T ** 3) * (r ** 2))

        return T_radiative

    def is_convective(self, r, rho, T, M, L):
        """
        Method to determine if the dominant temperature contributor at the inputted radius value is
        convective forces.

        :param r: The radius value at which the temperature contributions are being surveyed.
        :param rho: The density at `r`.
        :param T: The temperature at `r`.
        :param M: The total mass contained within `r`.
        :param L: The cumulative luminosity at `r`.
        :return: True if the dominant temperature contributor is convection, otherwise False.
        """

        # get the convective temperature value at `r`
        T_conv_value = self.temp_convective(r=r, rho=rho, T=T, M=M)

        # get the radiative temperature value at `r`
        T_rad_value = self.temp_radiative(r=r, rho=rho, T=T, L=L)

        is_conv_bool = -1.0 * T_conv_value >= -1.0 * T_rad_value

        return is_conv_bool

    # ---------- # DIFFERENTIAL EQUATIONS # ---------- #

    def hydrostat_equil(self, r, rho, T, M, L):
        """
        Method to define the hydrostatic equilibrium differential equation:
            drho/dr = - ((G * M * rho / r**2) + (dP/dT * dT/dr)) / (dP / drho)
        Note: The pressure-density and pressure-temperature differentials are calculated by
            differentiating the total pressure equation included in the `total_pressure` docstring
            as the sum of the non-relativistic degenerate pressure, the ideal gas pressure, and
            the photon gas pressure.

        :param r: The radius value for which the density, temperature, cumulative mass, and
            cumulative luminosity correspond.
        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param M: The cumulative mass contained within a radius, `r`.
        :param L: The cumulative luminosity radiated by the star, but considering contributions
            from inside a radius, `r`.
        :returns: The density derivative with respect to the radial component.
        """

        # calculate the pressure-temperature differential
        dP_dT = (rho * k / (self.mean_molec_weight() * m_p)) + (4 * a * (T ** 3) / 3)

        # calculate the pressure density differential
        deg_pressure_diff = (
            (3 * (numpy.pi ** 2)) ** (2/3)
        ) * (
            h_bar ** 2
        ) * (
            (rho / m_p) ** (2/3)
        ) / (
            3 * m_e * m_p
        )
        ideal_pressure_diff = (k * T) / (self.mean_molec_weight() * m_p)
        dP_drho = deg_pressure_diff + ideal_pressure_diff

        # calculate the temperature-radius differential
        dT_dr = self.temperature_grad(r=r, rho=rho, T=T, M=M, L=L)

        # calculate the gravitational component
        grav_comp = G * M * (1 + (self.lambda_small / r) + (r / self.lambda_large)) / (r ** 2)

        # calculate the density gradient
        density_gradient = -1.0 * ((grav_comp * rho) + (dP_dT * dT_dr)) / (dP_drho)

        return density_gradient

    def temperature_grad(self, r, rho, T, M, L):
        """
        Method to define the temperature gradient differential equation:
            dT/dr = - min(radiative temperature, convective temperature)

        :param r: The radius value for which the density, temperature, cumulative mass, and
            cumulative luminosity correspond.
        :param rho: The stellar density.
        :param T: The stellar temperature.
        :param M: The cumulative mass contained within a radius, `r`.
        :param L: The cumulative luminosity radiated by the star, but considering contributions
            from inside a radius, `r`.
        :returns: The temperature derivative with respect to the radial component.
        """

        # calculate the radiative temperature gradient
        T_radiative = self.temp_radiative(r=r, rho=rho, T=T, L=L)

        # calculate the convective temperature gradient
        T_convective = self.temp_convective(r=r, rho=rho, T=T, M=M)

        # calculate the temperature gradient
        temp_gradient = -1.0 * min(T_convective, T_radiative)

        return temp_gradient

    @staticmethod
    def mass_continuity(r, rho):
        """
        Method to define the mass continuity differential equation:
            dM/dr = 4 * pi * r**2 * rho

        :param r: The radius value for which the density correspond.
        :param rho: The stellar density.
        :returns: The mass derivative with respect to the radial component.
        """

        # calculate the mass gradient
        mass_gradient = 4 * numpy.pi * rho * (r ** 2)

        return mass_gradient

    def energy_equation(self, r, rho, T, energy_gen=None):
        """
        Method to define the energy differential equation:
            dL/dr = 4 * pi * r**2 * rho * epsilon

        :param r: The radius value for which the density and temperature correspond.
        :param rho: The stellar density.
        :param T: The stellar temperature.
        :returns: The luminosity derivative with respect to the radial component.
        """

        # obtain the energy generation rate necessary for defining the stellar luminosity;
        # if None is provided, use the total energy generation rate
        if energy_gen is None:
            energy_gen = self.energy_gen_rate(rho=rho, T=T, X_cno=None)

        # calculate the luminosity gradient
        lumin_gradient = 4 * numpy.pi * rho * energy_gen * (r ** 2)

        return lumin_gradient

    def optical_depth_equation(self, rho, T):
        """
        Method to define the optical depth differential equation:
            dtau/dr = kappa * rho

        :param rho: The stellar density.
        :param T: The stellar temperature.
        :returns: The density derivative with respect to the radial component.
        """

        # calculate the optical depth differential
        opt_depth_gradient = self.mean_opacity(rho=rho, T=T) * rho

        return opt_depth_gradient
