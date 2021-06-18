import setuptools
import os.path

from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bmodel",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"beattie_model_sensitivities/src"},
    python_requires=">=3.6",
    reqiures=['matplotlib, pints, scipy, symengine'],
    setup_requires=['pytest-runner', 'matplotlib', 'pints', 'scipy', 'symengine'],
    tests_require=['pytest']
)
