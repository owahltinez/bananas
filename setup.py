#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    'dataclasses', 'numpy', 'tqdm'
]

test_requirements = []

setup(
    name='bananas',
    version='0.0.1',
    description='Framework for ML libraries',
    long_description='Framework for ML libraries',
    author='omtinez',
    author_email='omtinez@gmail.com',
    url='https://gitlab.com/omtinez/bananas',
    packages=find_packages(),
    package_data={'': ['../README.md', '**/*.csv']},
    include_package_data=True,
    install_requires=requirements,
    license='MIT',
    zip_safe=False,
    keywords=['ML'],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
