# coding=utf-8
from setuptools import setup, find_packages
import pathlib
import pkg_resources


with pathlib.Path('requirements.txt').open() as requirements_txt:
    requirements = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
packages = find_packages(
    exclude=["​*.tests", "*​.tests.​*", "tests.*​", "tests"])

# magic function for including subpackages in repo
# can list packages with subpackages explicitly later
setup(
    name='iewIndex',
    version='0.0.1',
    packages=packages,
    license='Proprietary',
    author='Yuqi Wang, Gulai Shen',
    author_email='yuqi.wang@duke.edu',
    description=(
        'Indoor Environmental Wellness Index'
    ),
    install_requires=requirements
)