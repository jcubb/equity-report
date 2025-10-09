from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="equity-factor-attribution",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive equity factor attribution and risk analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/equity-report",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "equity-report=factor_reg:main",
        ],
    },
    keywords="finance, portfolio, attribution, risk, factor, regression, equity",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/equity-report/issues",
        "Source": "https://github.com/yourusername/equity-report/",
        "Documentation": "https://github.com/yourusername/equity-report/blob/main/README.md",
    },
)