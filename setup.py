import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="covid_simulation",
    use_scm_version=True,
    # version="0.4.3",
    author="UCA Datalab",
    author_email="datalab@uca.es",
    description="Predict future hospital admissions and simulate the number of beds occupied.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCA-Datalab/covid_occupancy_simulation.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.toml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, <=3.8.2',
    setup_requires=['setuptools_scm'],
    install_requires=[
        'cycler==0.10.0',
        'ezodf==0.3.2',
        'kiwisolver==1.2.0',
        'lxml==4.5.0',
        'matplotlib==3.2.1',
        'numpy==1.18.2',
        'pandas==1.0.3',
        'pyparsing==2.4.7',
        'python-dateutil==2.8.1',
        'pytz==2019.3',
        'scipy==1.10.0',
        'six==1.14.0'
    ]
)

