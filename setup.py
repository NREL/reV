"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
import sys
from warnings import warn

py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("reV is not compatible with python 2!")
elif py_version.minor < 6:
    warn("You will the get best results by running reV with python >= 3.6")

here = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

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

setup(
    name="reV",
    version=version,
    description="Renewable Energy Potential",
    long_description=readme,
    author="Galen Maclaurin",
    author_email="galen.maclaurin@nrel.gov",
    url="https://github.com/NREL/reV",
    packages=find_packages(),
    package_dir={"rev": "rev"},
    entry_points={
        "console_scripts": ["reV=reV.cli:main",
                            "reV-batch=reV.batch.cli_batch:main",
                            "reV-collect=reV.handlers.cli_collect:main",
                            "reV-econ=reV.econ.cli_econ:main",
                            "reV-gen=reV.generation.cli_gen:main",
                            "reV-multiyear=reV.handlers.cli_multi_year:main",
                            "reV-pipeline=reV.pipeline.cli_pipeline:main",
                            ("reV-aggregation=reV.supply_curve."
                             "cli_aggregation:main"),
                            ("reV-supply-curve=reV.supply_curve."
                             "cli_supply_curve:main"),
                            "reV-offshore=reV.offshore.cli_offshore:main",
                            ("reV-rep-profiles=reV.rep_profiles."
                             "cli_rep_profiles:main"),
                            ],
    },
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="rev",
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
