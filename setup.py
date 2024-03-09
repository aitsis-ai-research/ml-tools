from setuptools import setup, find_packages


setup(
    name='ml',
    version='0.1',
    description='Machine Learning Library',
    url='https://github.com/8Bits-ai/ml-tools',
    author='8Bits',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Tell distutils packages are under src

    install_requires=[
        'numpy',
        'typing'
            ],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)