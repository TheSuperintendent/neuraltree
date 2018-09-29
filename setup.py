from setuptools import setup, find_packages


setup(
    name="neuraltree",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    setup_requires=["Cython"],
    install_requires=[
        "Cython",
        "Keras",
        "pytest",
        "tensorflow"
    ],
    test_suite="test"
)
