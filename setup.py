from setuptools import setup, find_packages

# Setting up
setup(
    name="edynamics",
    version='0.1.3',
    author="Patrick Mahon",
    author_email="<pmahon3@uwo.ca>",
    description='Empirical dynamic modelling',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn'],
    keywords=['python', 'edm', 'time series', 'forecasting', 'empirical dynamics'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_dir={'edynamics': 'src'})
