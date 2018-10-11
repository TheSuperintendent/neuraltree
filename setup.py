from setuptools import setup, find_packages


setup(
    name="neuraltree",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    install_requires=[
        "Cython",
        "Keras",
        "pytest",
        "python-igraph",
        "tensorflow"
    ],
    test_suite="test"
)
