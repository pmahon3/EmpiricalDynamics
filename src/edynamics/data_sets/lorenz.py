import pandas as pd
import numpy as np


def _lorenz_integrator(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def lorenz_data() -> pd.DataFrame:
    dt = 0.01
    num_steps = 10000

    data = np.empty((num_steps + 1, 3))
    data[0] = (0., 1., 1.05)

    for i in range(num_steps):
        data[i + 1] = data[i] + _lorenz_integrator(data[i]) * dt

    index = pd.date_range(start=pd.Timestamp.now().round(freq='H'), periods=num_steps + 1, freq='H')
    data = pd.DataFrame(data=data, index=index, columns=['X', 'Y', 'Z'])

    return data
