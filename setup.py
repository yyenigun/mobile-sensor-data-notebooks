import os

from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='mobile-sensor-data-notebooks',
    version='1-SNAPSHOT',
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'future'],
    description='activity recognition python module',
    packages=find_packages(exclude='tests'),
    install_requires=[
        'h2o==3.10.4.6',
        'pymysql',
        'pandas',
        'numpy'
    ]
)
