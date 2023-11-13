from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ebs",
    version="0.5",
    author="Pin Chen",
    author_email="pin.chen@jpl.nasa.gov",
    description="Software for running error budget calculations for HWO using EXOSIMS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "run_ebs=ebs.cli_interface:main"
        ],
    },
)
