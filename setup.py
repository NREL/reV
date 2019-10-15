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

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "rst", format="md")

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


def get_sam_data_files():
    """
    Get all data files in the SAM directory that are required to run reV.
    """

    rel_dir = 'reV/SAM/'
    sam_dir = os.path.join(os.getcwd(), rel_dir)
    sam_data_files = []
    for root, _, files in os.walk(sam_dir):

        f_dir = root[root.index(rel_dir):]
        for fname in files:
            sam_data_files.append(os.path.join(f_dir, fname))

    sam_data_files = [f for f in sam_data_files
                      if not f.endswith(('.py', '.pyc'))]

    return sam_data_files


test_requires = ["pytest", ]

numpy_dependency = "numpy>=1.15.0"
pandas_dependency = "pandas>=0.23.0"
click_dependency = "click>=7.0"
scipy_dependency = "scipy>=1.2.1"

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
                            "reV-multiyear=reV.handerlers.cli_multi_year:main",
                            "reV-pipeline=reV.pipeline.cli_pipeline:main",
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
    ],
    test_suite="tests",
    install_requires=["click", "h5py", "numpy", "pandas", "psutil",
                      "rasterio", "scipy", "xarray", "NREL-PySAM"],
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["pypandoc", "flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
    data_files=[('sam_files', get_sam_data_files())],
)
