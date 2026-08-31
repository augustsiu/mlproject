from setuptools import setup, find_packages, setup


setup(
    name='mlproject',
    version='0.0.1',
    author='August',
    author_email='augustsiu@gmail.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn','seaborn'],
)