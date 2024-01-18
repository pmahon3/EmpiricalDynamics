"""
This module defines the functions for generating various data sets used in the testing process.

Please note, all data sets should be structured as pandas data frames, indexed by a datetime index.
"""
import os
import pytest
import pytest_html
import base64
import pandas as pd
import numpy as np
import logging

from scipy.integrate import odeint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set up directories for reports and plotting
logger.info("Setting up image directory for report plotting...")
os.environ['PYTEST_REPORT_IMAGES'] = os.path.dirname(os.path.realpath(__file__)) + "/report/images/"
os.makedirs(os.environ['PYTEST_REPORT_IMAGES'], exist_ok=True)

if os.listdir(os.environ['PYTEST_REPORT_IMAGES']):
    logger.info("Removing old images...")
    for f in os.listdir(os.environ['PYTEST_REPORT_IMAGES']):
        os.remove(os.path.join(os.environ['PYTEST_REPORT_IMAGES'], f))


# Define data sets
def generate_lorenz(n_points):
    """
    Generates a lorenz system data set.

    :param int n_points: the number of points to integrate
    :return pd.DataFrame: the integrated data.
    """

    def lorenz_system(state, t, sigma, beta, rho):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    # Lorenz system parameters
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = 28.0

    # Initial state (near the origin, but not exactly at it)
    initial_state = [1.0, 1.0, 1.0]

    # Time points at which to solve the system
    t = np.linspace(0, 50, n_points)  # for example, 10000 points up to t=50

    # Solve the system
    solution = odeint(lorenz_system, initial_state, t, args=(sigma, beta, rho))

    # Convert the solution to a pandas DataFrame
    data = pd.DataFrame(solution, columns=['X', 'Y', 'Z'])

    # Get the current datetime
    current_datetime = pd.Timestamp.now().round(freq='S')

    # Generate a datetime index starting from the current time with a frequency of one second
    datetime_index = pd.date_range(start=current_datetime, periods=len(data), freq='S')

    # Replace the existing index with the new datetime index
    data.index = datetime_index

    return data


def generate_duffing_oscillator(n_points):
    """
    Generates a duffing oscillator system data set.

    :param int n_points: the number of points to integrate
    :return pd.DataFrame: the integrated data.
    """

    def duffing_system(state, t, alpha, beta, delta, gamma, omega):
        x, y = state
        dx_dt = y
        dy_dt = x - beta * x ** 3 - delta * y + gamma * np.cos(omega * t)
        return [dx_dt, dy_dt]

    # Duffing oscillator parameters
    alpha = 1.0
    beta = 1.0
    delta = 0.3
    gamma = 0.37
    omega = 1.2

    # Initial state
    initial_state = [0.1, 0.0]

    # Time points at which to solve the system
    t = np.linspace(0, 50, n_points)

    # Solve the system
    solution = odeint(duffing_system, initial_state, t, args=(alpha, beta, delta, gamma, omega))

    # Convert the solution to a pandas DataFrame
    data = pd.DataFrame(solution, columns=['X', 'Y'])

    # Get the current datetime
    current_datetime = pd.Timestamp.now().round(freq='S')

    # Generate a datetime index starting from the current time with a frequency of one second
    datetime_index = pd.date_range(start=current_datetime, periods=len(data), freq='S')

    # Replace the existing index with the new datetime index
    data.index = datetime_index

    return data


@pytest.fixture(scope='module')
def data_set(request):
    """
    A data set generating factory used to generate pre-defined data sets.

    :param request: the data set request structured as an indirect pytest parametrization.
    :return pd.DataFrame: the generated data set.
    """
    params = request.param
    dataset_type = params['type']
    n_points = params['n_points']

    if dataset_type == 'lorenz':
        return generate_lorenz(n_points=n_points)
    elif dataset_type == 'duffing':
        return generate_duffing_oscillator(n_points=n_points)
    else:
        raise ValueError("Unknown dataset type")


# Define a fixture to initialize image paths
@pytest.fixture
def image_paths(request):
    # Initialize an empty list for image paths
    request.node.image_paths = []

    # Return the list, so it can be manipulated within the test
    return request.node.image_paths


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()

    # Correctly initialize 'extra' if it's not already an attribute of report
    if not hasattr(report, 'extra'):
        report.extra = []

    # Assuming you have set the 'image_paths' attribute in your test
    for image_path in getattr(item, 'image_paths', []):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            html = f'<div><img src="data:image/png;base64,{encoded_string}" alt="image" style="width:1200px; height:auto;"/></div>'
            report.extra.append(pytest_html.extras.image(encoded_string, mime_type='image/png'))
