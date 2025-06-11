from setuptools import setup, find_packages

setup(
    name='windtunnel',
    version='0.1.0',
    description='Retrieval-augmented conditional diffusion market simulator',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    url='https://github.com/jialuechen/windtunnel',
    license='MIT',
    packages=find_packages(exclude=('tests', 'scripts')), 
    install_requires=[
        'torch>=1.10',
        'numpy',
        'pandas',
        'scipy',
        'fastdtw',
        'scikit-learn'
    ],
    extras_require={
        'dev': ['pytest', 'black', 'flake8']
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
