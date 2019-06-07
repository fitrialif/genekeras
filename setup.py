from setuptools import setup

setup(
    name='genekeras',
    version='0.0.1',
    description='My private package from private github repo',
    url='https://github.com/d-corsi/GeneKeras',
    author='Davide Corsi',
    author_email='corsi.davide@outlook.com',
    license='unlicense',
    packages=['numpy'],
    install_requires=[],
    packages=find_packages()
)