from setuptools import setup, find_packages
requirements = [
    "numpy>=1.24.2",
    "pandas>=1.5.3",
    "ray>=2.2.0",
    "scipy>=1.8.0",
    "setuptools>=57.0.0",
    "tqdm>=4.64.1"
]

# Setting up
setup(
    name="edynamics",
    version='0.3.12',
    author="Patrick Mahon",
    author_email="<pmahon3@uwo.ca>",
    description='Empirical dynamic modelling - modular, parallel, object-oriented',
    packages=find_packages(where="src"),
    install_requires=requirements,
    keywords=['python', 'edm', 'time series', 'forecasting', 'empirical dynamics'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_dir={"": "src"})
