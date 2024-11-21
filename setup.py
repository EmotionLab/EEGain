from pathlib import Path

from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory

README = (PARENT / "README.md").read_text(encoding="utf-8")

with open('requirements.txt') as f:
    required = f.read().splitlines()

exec(open(str(PARENT / "eegain" / "version.py")).read())
setup(
    name="eegain",
    version=__version__,
    description="EEG emotion recognition package for standardization",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/RRaphaell/EEGain",
    author="GAIN Team",
    packages=find_packages(),
    author_email="",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    keywords="EEG, emotion recognition, framework, package, standardized",
    install_requires=required,
    python_requires=">=3.7",
)
