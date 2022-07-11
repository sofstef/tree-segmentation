from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Sofija Stefanovic",
    author_email="ss2536@cam.ac.uk",
    description="Tree trunk segmentation with depth data from smartphones and a modified UNet.",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
