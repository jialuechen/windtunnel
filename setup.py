from setuptools import setup, find_packages

setup(
    name='windtunnel',
    version='1.0.0',  # Incremented version to reflect new features
    description='A Multi-Layer Generative Market Simulation Framework',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    packages=find_packages(),
    install_requires=[
        'openai>=0.27.0',  # For NLP-based configuration generation
        'plotly>=5.0.0',   # For interactive visualization
    ],
    python_requires='>=3.7',
)