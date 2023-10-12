from setuptools import setup, find_packages


# Requirements
with open("epgpy/version.py") as version_file:
    exec(version_file.read())

setup(
    name="epgpy",
    version=__version__,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy"],
    extras_require={
        "full": ["scipy", "cupy", "matplotlib"],
        "examples": "click",
        "test": "pytest",
    },
    # metadata for upload to PyPI
    description="A Python library implementing the Extended-Phase Graph (EPG) algorithm and extensions",
    license="",
    # url=,
)
