from setuptools import setup

setup(
    name="paramopt",
    version="0.1.0",
    license="MIT",
    description="Library for easy parameter tuning using gaussian process regression",
    author="Kota Aono",
    url="https://github.com/ut-hnl-lab/paramopt.git",
    packages=['paramopt'],
    install_requires=[
        'matplotlib',
        'numpy',
        'GPy',
        'GPyOpt',
        'sklearn',
        'scipy'
    ]
)
