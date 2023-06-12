from setuptools import find_packages, setup

setup(
    name='Gaussianization-Bound',
    description='Gaussianization Convergence Rate Bound',
    version='0.1dev',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    author='Felix Draxler, Lars Kühmichl',
    author_email='felix.draxler@iwr.uni-heidelberg.de, lars.kuehmichl@iwr.uni-heidelberg.de',
    url='https://github.com/vislearn/Gaussianization-Bound'
)
