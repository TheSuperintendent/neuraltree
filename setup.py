from setuptools import setup, find_packages


setup(
    name="neuraltree",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    install_requires=[
        "Cython",
        "Keras",
        "pytest",
        # TODO: implement custom install command to install the igraph C core
        "python-igraph",
        "tensorflow"
    ],
    test_suite="test"
)
