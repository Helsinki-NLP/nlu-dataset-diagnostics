from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'google-cloud-storage>=1.14.0',
    'transformers',
    'datasets',
    'numpy==1.18.5',
    'argparse',
    'tqdm==4.49.0',
    'nltk'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='NLU Dataset Diagnostics'
)
