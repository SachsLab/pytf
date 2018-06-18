import __future__
from setuptools import setup
import os

# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : MIT

setup(
    name = "pytf",
    version = 0.1,
    packages = ['pytf', 'pytf.viz', 'pytf.viz.utils', 'pytf.filter', 'pytf.reconstruction', 'pytf.time_frequency', 'pytf.utilities'],
    install_requires=['numpy', 'scipy'],
    author = "David Lu",
    author_email = "davidlu89@gmail.com",
    description = "pytf is a tool for time-frequency analysis",
    url='https://github.com/davidlu89/pytf',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
        platforms='any'
)
