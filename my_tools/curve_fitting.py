# curve_fitting.py
# Curve fitting, empirical models, and analytical functions
# Originally used in: Yale ECON 331: Economics of Energy and Climate Change (polynomial fitting),
#                     Research in conjuction with work at Ellington Management Group (Black-Scholes option pricing),
#                     Yale MENG 363L: Fluids Dynamics Lab (viscosity correlations)

import numpy as np
from scipy.stats import norm
import math


def poly_fit_report(x, y, degrees=[1, 2, 3]):
    """Fit polynomials of multiple degrees and report coefficients and residuals.

    Args:
        x: array-like of independent variable values
        y: array-like of dependent variable values
        degrees: list of polynomial degrees to try

    Returns:
        dict mapping degree -> {'coeffs': array, 'residuals': float, 'poly': np.poly1d}
    """
    results = {}
    for deg in degrees:
        coeffs, residuals, _, _, _ = np.polyfit(x, y, deg, full=True)
        poly = np.poly1d(coeffs)
        res_val = residuals[0] if len(residuals) > 0 else None
        results[deg] = {
            'coeffs': coeffs,
            'residuals': res_val,
            'poly': poly
        }
    return results


def d1(S, K, T, r, q, sigma):
    """Black-Scholes d1 parameter.

    Args:
        S: spot price
        K: strike price
        T: time to expiration
        r: risk-free rate
        q: dividend yield
        sigma: volatility

    Returns:
        d1 value
    """
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))


def d2(S, K, T, r, q, sigma):
    """Black-Scholes d2 parameter.

    Args:
        S, K, T, r, q, sigma: same as d1

    Returns:
        d2 value
    """
    return d1(S, K, T, r, q, sigma) - sigma*np.sqrt(T)


def black_scholes_call(S, K, T, r, q, sigma):
    """Black-Scholes European call option price.

    Args:
        S: spot price
        K: strike price
        T: time to expiration (years)
        r: risk-free rate
        q: continuous dividend yield
        sigma: volatility

    Returns:
        call option price
    """
    return S*np.exp(-q*T)*norm.cdf(d1(S, K, T, r, q, sigma)) - \
           K*np.exp(-r*T)*norm.cdf(d2(S, K, T, r, q, sigma))


def sutherland_viscosity(T_celsius):
    """Dynamic viscosity of air using Sutherland's formula.

    mu = 1.458e-6 * T^0.5 / (1 + 110.4/T) where T is in Kelvin.

    Args:
        T_celsius: temperature in Celsius (scalar or array)

    Returns:
        dynamic viscosity in Pa*s
    """
    T_kelvin = T_celsius + 273.15
    return 1.458e-6 * pow(T_kelvin, 0.5) / (1 + 110.4 / T_kelvin)


def sutherland_viscosity_error(T_celsius, T_uncertainty):
    """Uncertainty in Sutherland viscosity due to temperature uncertainty.

    Propagated analytically from the Sutherland formula.

    Args:
        T_celsius: temperature in Celsius
        T_uncertainty: uncertainty in temperature (Celsius)

    Returns:
        uncertainty in dynamic viscosity
    """
    T_kelvin = T_celsius + 273.15
    return 1.458e-6 * pow(T_kelvin, 0.5) * \
           (3*110.4 + T_kelvin) / 2 / pow(110.4 + T_kelvin, 2) * T_uncertainty


def andrade_viscosity(T_celsius):
    """Dynamic viscosity of water using the Andrade equation.

    mu = 2.414e-5 * exp(247.8 / (T - 140)) where T is in Kelvin.

    Args:
        T_celsius: temperature in Celsius (scalar or array)

    Returns:
        dynamic viscosity in Pa*s
    """
    T_kelvin = T_celsius + 273.15
    return 2.414e-5 * math.exp(247.8 / (T_kelvin - 140))


def andrade_viscosity_error(T_celsius, T_uncertainty):
    """Uncertainty in Andrade viscosity due to temperature uncertainty.

    Args:
        T_celsius: temperature in Celsius
        T_uncertainty: uncertainty in temperature (Celsius)

    Returns:
        uncertainty in dynamic viscosity
    """
    T_kelvin = T_celsius + 273.15
    return 2.414e-5 * math.exp(247.8 / (T_kelvin - 140)) * \
           247.8 / (T_kelvin - 140) / (T_kelvin - 140) * T_uncertainty
