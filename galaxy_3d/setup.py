import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galaxy_3d", # Replace with your own username
    version="0.0.1",
    author="Chen Heinrich",
    author_email="chenhe@caltech.edu",
    description="Theory code for 3D galaxy power spectrum and bispectrum",
    long_description_content_type="text/markdown",
    #url="https://github.com/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.9',
)