from setuptools import setup

import setuptools

setup(
    name='app',
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'Flask',
        'numpy',
        'pandas',
        'bokeh'
        ]
)