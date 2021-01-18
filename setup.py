import setuptools

VERSION = '0.0.1'

setuptools.setup(
    name="rapid",
    version=VERSION,
    author="Daochen Zha",
    author_email="daochen.zha@tamu.edu",
    description="Source core of RAPID",
    long_description='RAPID is simple method for exploration',
    long_description_content_type="text/markdown",
    url="https://github.com/daochenzha/rapid",
    keywords=["Exploreration", "Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.5',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
