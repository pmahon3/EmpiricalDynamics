from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.2'
DESCRIPTION = 'Empirical dynamic modelling'
LONG_DESCRIPTION = 'Nonlinear state space reconstruction methods for time series analysis and forecasting'

# Setting up
setup(
    name="edynamics",
    version=VERSION,
    author="Patrick Mahon",
    author_email="<pmahon3@uwo.ca>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn'],
    keywords=['python', 'edm', 'time series', 'forecasting', 'empirical dynamics'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Researchers/Data Scientists",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_dir={'': 'src'})
