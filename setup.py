from setuptools import setup, find_packages

setup(
    name='topofisher',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'numpy',
        'scipy',
        'gudhi',
        'ripser',
        'tqdm',
        'pyyaml',
        'powerbox',
    ],
    extras_require={
        'gnn': [
            'torch-geometric',
        ],
        'mma': [
            'multipers',
        ],
        'lensing': [
            'jax',
            'jax-cosmo',
        ],
        'analysis': [
            'pandas',
            'matplotlib',
        ],
        'all': [
            'torch-geometric',
            'multipers',
            'jax',
            'jax-cosmo',
            'pandas',
            'matplotlib',
        ],
    },
    python_requires='>=3.8',
    author='Matteo Biagetti, Mathieu Carrière, Francesco Conti, Enrico Maria Ferrari, Sven C. Heydenreich, Karthik Viswanathan',
    description='Learning topological summary statistics by maximizing Fisher information',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
