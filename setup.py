import os
import re

from io import open
from setuptools import setup


def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        return f.read()

def get_requirements():
    with open('requirements.txt', encoding="utf-8-sig") as f:
        return f.read().splitlines()

def get_version():
    with open(os.path.join("craft", "__init__.py"), encoding="utf-8-sig") as f:
        return re.search(r'^__version__[ ]*=[ ]*([\'"])([^\'"]+)\1', f.read(), re.M).group(2)

setup(
    name="craft-detector",
    version=get_version(),
    author="Clova AI Research, NAVER Corp.",
    license="MIT",
    author_email="youngmin.baek@navercorp.com",
    description="Python package for Official implementation of Character Region Awareness for Text Detection (CRAFT)",
    maintainer="Won-Kyu Park",
    maintainer_email="wkpark@gmail.com",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wkpark/CRAFT-pytorch",
    package_dir={
        "craft_detector": "craft",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    install_requires=get_requirements(),
    python_requires='>=3.6',
    keywords="machine-learning, deep-learning, ml, pytorch, text, text-detection, craft",
)
