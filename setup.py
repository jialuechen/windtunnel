from setuptools import setup, find_packages

setup(
    name='windtunnel',
    version='0.1.0',
    description='Retrieval-augmented conditional diffusion market simulator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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
        'scikit-learn',
        'plotly',  
        'openai', 
    ],
    extras_require={
        'dev': ['pytest', 'black', 'flake8', 'build', 'twine']
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
