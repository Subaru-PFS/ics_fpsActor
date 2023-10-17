import distutils
from distutils.core import setup, Extension

try:
    from sdss3tools import setup
except ImportError:
    from setuptools import setup

import os

setup(name="ics_fpsActor",
      description= "Toy SDSS-3 actor.",
      packages=["ics.fpsActor"],
      package_dir={'': 'python'},
     )

