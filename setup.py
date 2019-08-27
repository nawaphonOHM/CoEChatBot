from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.17.0", "tensorflow>=1.11.0", "pythainlp>=2.0.6", "pandas>=0.25.0", "keras>=2.2.5", "wheel>=0.33.6"]

setup(
    name="run_model",
    version="2.0.0",
    author="Nawaphon Isarathanachaikul",
    author_email="nawaphon2539@gmail.com",
    description="A Project",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nawaphonOHM/CoEChatBot",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Machine Learning"
    ],
)