from setuptools import setup, find_packages

setup(
    name="machine",
    url="https://github.com/mmeehan/machine",
    author="MichaelMeehan",
    author_email="michael.meehan1@jcu.edu.au",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    version="0.1",
    license="MIT",
    description="Blah.",
)
