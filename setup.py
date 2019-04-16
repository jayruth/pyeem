from setuptools import setup
import os
from setuptools import find_packages

with open(os.path.join('.', 'README.md')) as file:
    long_description = file.read()

opts = dict(
            name="pyeem",
            version="0.1",
            packages=find_packages(),
            install_requires=['numpy',
                              'pandas',
                              'sklearn',
                              'h5py',
                              'xlrd'],
            author="Jay Rutherford, Ben Ponto, and Neal Dawson-Elli",
            description="A machine learning model to test for presence of smoke sources",
            long_description=long_description,
            license="MIT",
            keywords="Deep Neural Networks",
            include_package_data=True)

if __name__ == "__main__":
    setup(**opts)