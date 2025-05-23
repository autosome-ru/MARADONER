__requires__ = ['pip >= 24.0']
from maradoner import __version__, __min_reqs__
from setuptools import setup, find_packages
import os

with open('README.md', 'r', encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='maradoner',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['data/*']},
    description='Variance-adjusted estimation of motif activities.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'maradoner = maradoner.main:main',
        ],
    },
    author='Georgy Meshcheryakov',
    author_email='iam@georgy.top',
    install_requires=__min_reqs__,
    python_requires='>=3.9',
    url="https://github.com/autosome-ru/nemara",
    classifiers=[
	      "Programming Language :: Python :: 3.9",
	      "Programming Language :: Python :: 3.10",
	      "Programming Language :: Python :: 3.11",
	      "Programming Language :: Python :: 3.12",
	      "Topic :: Scientific/Engineering",
              "Operating System :: OS Independent"]

)