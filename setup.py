#
# MarkovModels setuptools script
#
import os

from setuptools import find_packages, setup


# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number
#with open(os.path.join('MarkovModels', 'version.txt'), 'r') as f:
 #   version = f.read()

version = "0"

# Go!
setup(
    # Module name (lowercase)
    name='MarkovModels',

    version=version,
    description='markov models for cardiac modelling',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Joseph Shuttleworth, Dominic Whittaker, Michael Clerx, Maurice Hendrix, Gary Mirams',
    author_email='joseph.shuttleworth@nottingham.ac.uk',
    maintainer='Joseph Shuttleworth',
    maintainer_email='joseph.shuttleworth@nottingham.ac.uk',
    url='https://github.com/joeyshuttleworth/MarkovModels',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # Packages to include
    packages=find_packages(
        include=('MarkovModels', 'MarkovModels.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.6',

    # List of dependencies
    install_requires=[
        'pints>=0.3',
        'scipy>=1.7',
        'numpy>=1.17',
        'matplotlib>=3.4',
        'pandas>=1.3',
        'sympy>=1.8',
        'numba>=0.54.1',
        'NumbaLSODA>=0.1.7',
        'regex>=2021.11.10',
        'myokit>=1.33.0',
        'seaborn>=0.12.0',
        'markov_builder @ git+ssh://git@github.com/CardiacModelling/Markov-builder@add_wang_model'
    ],
    extras_require={
        'test': [
            'pytest-cov>=2.10',     # For coverage checking
            'pytest>=4.6',          # For unit tests
            'flake8>=3',            # For code style checking
            'isort',
            'mock>=3.0.5',         # For mocking command line args etc.
            'codecov>=2.1.3',
        ],
    },
)
