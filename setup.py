from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="big-data-model",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A big data processing model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/big-data-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.3",
        "scikit-learn>=0.24.2",
        "joblib>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "run-big-data-model=main:main",
        ],
    },
)
