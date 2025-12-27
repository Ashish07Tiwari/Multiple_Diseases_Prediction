from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT= "-e ."

def get_reuirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name= "Multiple Diseases Prediction",
    version= "0.0.1",
    author= "Ashish Tiwari",
    author_email= "ashishddndt@gmail.com",
    packages= find_packages(),
    install_requirements= get_reuirements('requirements.txt')
)
