from setuptools import setup, find_packages

setup(
    name="prism",
    version="0.1.0",
    description="Short description of Prism",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Supraja Chittari",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "lmfit",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "scipy",
        "setuptools",
        "openpyxl",
    ],
)
