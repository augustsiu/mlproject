from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Read the requirements file and return a list of requirements.
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        # Skip blank lines, comments, and the editable-install marker
        requirements = [
            req for req in requirements
            if req and not req.startswith("#") and req != HYPHEN_E_DOT
        ]

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='August',
    author_email='augustsiu@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)