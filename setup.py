from setuptools import setup, find_packages

setup(name='EmpiricalDynamics',
      version='0.1.2',
      description='Modelling tools for nonlinear state space reconstruction analysis',
      packages=['edynamics',
                'edynamics.modelling_tools',
                'edynamics.modelling_tools.blocks',
                'edynamics.modelling_tools.data_types',
                'edynamics.modelling_tools.models'],
      author='Patrick Mahon',
      author_email='pmahon3@uwo.ca',
      zip_safe=False)
