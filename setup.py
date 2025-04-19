from setuptools import find_packages, setup

setup(
    name="ollama-mcp-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-core>=0.1.30",
        "langchain>=0.1.12",
        "requests>=2.31.0",
        "pydantic>=2.6.3",
        "typing-extensions>=4.10.0",
        "setuptools>=69.1.1",
        "wheel>=0.42.0",
    ],
    author="godstale",
    author_email="godstale@hotmail.com",
    description="A library for using Ollama with MCP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/godstale/ollama-mcp-agent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
