from setuptools import setup

setup(
    name="paramopt",
    version="0.1.0",
    license="MIT",
    description="Measurement package for Keyence laser profilers",
    author="KotaAono",
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
