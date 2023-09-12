#
# markovmodels setuptools script
#

from setuptools import find_packages, setup

# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number with open(os.path.join('markovmodels', 'version.txt'),
# 'r') as f: version = f.read()

version = "0.1.0"

# Go!
setup(
    # Module name (lowercase)
    name='markovmodels',

    version=version,
    description='markov models for cardiac modelling',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Joseph Shuttleworth, Dominic Whittaker, Michael Clerx, Maurice Hendrix, Gary Mirams',
    author_email='joseph.shuttleworth@nottingham.ac.uk',
    maintainer='Joseph Shuttleworth',
    maintainer_email='joseph.shuttleworth@nottingham.ac.uk',
    url='https://github.com/joeyshuttleworth/markovmodels',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # Packages to include
    packages=find_packages(
        include=('markovmodels', 'markovmodels.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.8',

    # List of dependencies
    install_requires=[
        'pints>=0.4.0',
        'scipy>=1.9.1',
        'numpy>=1.23.3',
        'matplotlib>=3.6.2',
        'pandas>=1.5.0',
        'sympy<=1.11.1',
        'numba>=0.56.2',
        'regex>=2022.9.13',
        'myokit>=1.33.0',
        'seaborn>=0.12.0',
        'markov_builder @ git+https://git@github.com/CardiacModelling/Markov-builder@thirty_models',
        'scikit-build>=0.16.7',
        'numbalsoda @ git+https://git@github.com/NicholasWogan/numbalsoda@main'
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
