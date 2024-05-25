import setuptools

setuptools.setup(
    name='shogunfolio',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        "shogun==6.1.4",
        "pandas==2.0.3",
        "networkx==3.1"
    ],
    url='https://github.com/jialuechen/shogunfolio',
    license='BSD-2',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    description='Portfolio Optimization Python Library Built on top of shogun'
)