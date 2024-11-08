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


def integrate_polynomial(bounds, coeffs):
    """Exact integral of a polynomial with given bounds and coefficients."""
    right = polynomial(bounds[1], coeffs)

# def integrate_naive(x, fn, f_args):
#     """Riemann sum approximation to integral."""
#     sum = 0
#     for i_x, x_ in enumerate(x):
#         if not i_x:
#             continue
#         y = fn(x_, **f_args)
#         delta_x = x[i_x] - x[i_x - 1]
#         sum += y * delta_x
    
#     return sum

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

print("Exact integral true function: ")
print(f"Numerical integration true function: {num_integral_true}")
print(f"Numerical integration noisy function: {num_integral_noisy}")

plt.figure("Function")
plt.plot(x, y_true, label="True", linestyle="--")
plt.plot(x, y_noisy, label="Noisy", alpha=0.5)
plt.legend()
plt.show()
