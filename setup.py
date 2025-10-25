import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KDAREK",
    version="0.0.1",
    author="Masoud Ataei",
    author_email="Masoud.Ataei@maine.edu",
    description="KDAREK : Kurkova Distance-Aware Errors for Kolmogorov Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/Masoud-Ataei/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={  
        'KDAREK': [            
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pykan>=0.2.8",
		"DAREK",
    ],
)
