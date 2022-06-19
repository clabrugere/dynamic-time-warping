from setuptools import setup

setup(
    name="dynamic-time-warping",
    version="0.1",
    description="Small module to efficiently compute dynamic time warping distances",
    url="",
    packages=["dtw"],
    install_requires=[
        "numpy>=1.21.5",
        "numba>=0.55.1",
    ],
    license="MIT",
    author="Clément Labrugere",
)
