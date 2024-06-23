from setuptools import setup, find_packages

setup(
    name='KANBert',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.3.1',
        'transformers>=4.41.2'
    ],
)
