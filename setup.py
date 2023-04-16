from setuptools import setup, find_packages
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

# Setting up
setup(
    name="edynamics",
    version='0.3.8',
    author="Patrick Mahon",
    author_email="<pmahon3@uwo.ca>",
    description='Empirical dynamic modelling - modular, parallel, object-oriented',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    keywords=['python', 'edm', 'time series', 'forecasting', 'empirical dynamics'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_dir={'edynamics': 'src'})
