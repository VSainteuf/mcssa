from setuptools import setup, find_packages

setup(name='mcssa',
      version="0.0.1",
      description='Univariate Monte Carlo Singular Spectrum Analysis',
      author='Vivien Sainte Fare Garnot',
      packages=find_packages(exclude=['contrib', 'docs', 'example']),
      include_package_data=True,
      python_requires='>=3.4')
