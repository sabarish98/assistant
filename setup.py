"""Setup configuration for AI Research Assistant."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-research-assistant",
    version="1.0.0",
    author="AI Research Assistant Team",
    author_email="",
    description="A comprehensive AI Research Assistant for document processing and semantic search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-research-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Database :: Database Engines/Servers",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "pre-commit>=3.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "research-assistant=interfaces.cli:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/yourusername/ai-research-assistant/blob/main/PROJECT_OVERVIEW.md",
        "Source": "https://github.com/yourusername/ai-research-assistant",
        "Tracker": "https://github.com/yourusername/ai-research-assistant/issues",
    },
    include_package_data=True,
    zip_safe=False,
)