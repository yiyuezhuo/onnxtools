# This file is just used to invoke `pip install -e .` or `python setup.py` to introduce a "editable" package into global.

from setuptools import setup, find_packages

requires = [
    "onnx",
    "numpy",
]


setup(
    name='onnxtools',
    version='0.0.0',
    author='yiyuezhuo',
    author_email='yiyuezhuo@gmail.com',
    install_requires=requires,
    packages=find_packages(),
)
