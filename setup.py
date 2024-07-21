import setuptools

setuptools.setup(
    name='deepfolio',
    version='1.0.0',
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow>=2.11.1",
        "keras>=3.0.0",
        "pandas==2.0.3",
        "networkx==3.1"
    ],
    url='https://github.com/jialuechen/deepfolio',
    license='BSD-2',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    description='Portfolio Optimization Python Library Built on top of Keras and Tensorflow'
)
