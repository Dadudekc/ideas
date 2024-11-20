from setuptools import setup, find_packages

setup(
    name="basicbot",
    version="0.1",
    description="A basic trading bot built with Python.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/basicbot",  # Replace with your repository URL
    packages=find_packages(),  # Automatically find packages in the project
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    install_requires=[
        "pandas>=1.3.0",
        "ta>=0.10.0",
        "backtrader>=1.9.78.123",
        "pytest>=7.0.0",  # Add dependencies used in your project
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "basicbot=basicbot.main:main",  # Replace with your entry point function
        ]
    },
)
