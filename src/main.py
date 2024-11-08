import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate


def polynomial(x, coeffs):
    """
    Calculate the polynomial with coefficients given by the list coeffs.
    """
    l_x = [coeff_ * x**deg for deg, coeff_ in enumerate(coeffs)]

    return sum(l_x)


def integrate_naive(x, y):
    """Riemann sum approximation to integral."""
    sum = 0
    for i_pt, y_ in enumerate(y):
        if not i_pt:
            continue
        delta_x = x[i_pt] - x[i_pt - 1]
        sum += y_ * delta_x
    
    return sum


def derivative_polynomial(x, coeffs):
    """Calculate the derivative of a polynomial at point x."""
    coeffs_mod = [coeff_ * i_coeff for i_coeff, coeff_ in enumerate(coeffs)][1:]
    
    return polynomial(x, coeffs_mod)


def integrate_polynomial(bounds, coeffs):
    """Exact integral of a polynomial with given bounds and coefficients."""
    coeffs_mod = [0]
    for i_coeff, coeff_ in enumerate(coeffs):
        coeffs_mod.append(coeff_ / (i_coeff + 1))
    
    return polynomial(bounds[1], coeffs_mod) - polynomial(bounds[0], coeffs_mod)


# def integrate_gauss_hermite(bounds, f, f_args, degree):
#     """"""


x_min = 0
x_max = 1
delta_x = 1e-3
x = np.arange(x_min, x_max , delta_x)

l_coeffs = [1, 1, 1]
f_args = {"coeffs": l_coeffs}
y_true = polynomial(x, l_coeffs)
rng = np.random.default_rng()
scale_noise = 1e-1
noise = rng.normal(loc=0, scale=scale_noise, size=len(y_true))
y_noisy = y_true + noise
num_integral_true = integrate_naive(x, y_true)
num_integral_noisy = integrate_naive(x, y_noisy)
exact_integral = integrate_polynomial((x_min, x_max), l_coeffs)
print(f"Exact integral true function: {exact_integral}")
print(f"Numerical integration true function: {num_integral_true}")
print(f"Numerical integration noisy function: {num_integral_noisy}")
error_num_true = exact_integral - num_integral_true
error_num_noisy = exact_integral - num_integral_noisy
print("Error between exact integral and numerical integration using " \
      f"noiseless samples: {error_num_true:.3e}")
print("Error between exact integral and numerical integration using " \
      f"noisy samples: {error_num_noisy:.3e}")
plt.figure("Function")
plt.plot(x, y_true, label="True", linestyle="--")
# plt.plot(x, derivative_polynomial(x, l_coeffs), label="Derivative", alpha=0.75)
plt.plot(x, y_noisy, label="Noisy", alpha=0.5)
plt.legend()
plt.show()
