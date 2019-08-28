from setuptools import setup, find_packages

requirements = None

with open("README.md", "r") as readme_file:
    readme = readme_file.read()
with open("./requirements.txt", "r") as requires:
    requirements = requires.readlines()

setup(
    name="CoEChatBotService",
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