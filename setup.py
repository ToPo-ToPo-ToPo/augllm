
# Author: Shun Ogawa (a.k.a. "ToPo")
# Copyright (c) 2025 Shun Ogawa (a.k.a. "ToPo")
# License: Apache License Version 2.0

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent

NAME = 'augllm'
AUTHOR = 'Shun Ogawa (a.k.a. "ToPo")'
DESCRIPTION = "A library for augmenting large language models",
LONG_DESCRIPTION = (this_directory / "README.md").read_text(encoding="utf-8")
URL = 'https://github.com/ToPo-ToPo-ToPo/augllm'
LICENSE = 'Apache License Version 2.0'
VERSION = '1.0'
PYTHON_REQUIRES = ">=3.11"
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
]

INSTALL_REQUIRES = [
    "pyyaml",
    "ollama",
    "PyMuPDF",
    "Pillow",
    "tqdm",
    "pytest",
]


setup(
    name=NAME,
    author=AUTHOR,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
)
