#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="bananas",
    version="0.0.1",
    description="Framework for machine learning libraries",
    long_description="Framework for machine learning libraries",
    author="owahltinez",
    author_email="oscar@wahltinez.org",
    url="https://github.com/owahltinez/bananas",
    packages=find_packages(),
    package_data={"": ["../README.md", "**/*.csv"]},
    include_package_data=True,
    install_requires=["numpy", "dataclasses", "tqdm"],
    license="MIT",
    zip_safe=False,
    keywords=["ML"],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    test_suite="tests",
    tests_require=[],
)
