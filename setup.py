"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn

here = os.path.abspath(os.path.dirname(__file__))

PYPI_DISALLOWED_RST = {"raw::", "<p", "</p", "<img", "---------"}
readme = []
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    for line in f.readlines():
        if any(substr in line for substr in PYPI_DISALLOWED_RST):
            continue
        readme.append(line)

readme = "".join(readme).replace(
    "A visual summary of this process is given below:", ""
)

with open(os.path.join(here, "reV", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


with open("requirements.txt") as f:
    install_requires = f.readlines()

test_requires = ["pytest>=5.2", ]
description = ("National Renewable Energy Laboratory's (NREL's) Renewable "
               "Energy Potential(V) Model: reV")

setup(
    name="NREL-reV",
    version=version,
    description=description,
    long_description=readme,
    author="Galen Maclaurin",
    author_email="galen.maclaurin@nrel.gov",
    url="https://nrel.github.io/reV/",
    packages=find_packages(),
    package_dir={"reV": "reV"},
    entry_points={
        "console_scripts": ["reV=reV.cli:main",
                            "reV-bespoke=reV.bespoke.cli_bespoke:main",
                            "reV-collect=reV.handlers.cli_collect:main",
                            "reV-econ=reV.econ.cli_econ:main",
                            "reV-gen=reV.generation.cli_gen:main",
                            "reV-multiyear=reV.handlers.cli_multi_year:main",
                            ("reV-supply-curve-aggregation=reV.supply_curve."
                             "cli_sc_aggregation:main"),
                            ("reV-supply-curve=reV.supply_curve."
                             "cli_supply_curve:main"),
                            "reV-nrwal=reV.nrwal.cli_nrwal:main",
                            ("reV-rep-profiles=reV.rep_profiles."
                             "cli_rep_profiles:main"),
                            "reV-QA-QC=reV.qa_qc.cli_qa_qc:main",
                            "reV-hybrids=reV.hybrids.cli_hybrids:main",
                            ("reV-project-points=reV.config."
                             "cli_project_points:project_points"),
                            ],
    },
    package_data={'reV': ['SAM/defaults/*.json', 'SAM/defaults/*.csv',
                          'generation/output_attributes/*.json']},
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="reV",
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
        "s3": ['fsspec', 's3fs'],
        "hsds": ["hsds>=0.8.4"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
