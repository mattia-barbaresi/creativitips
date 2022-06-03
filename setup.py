from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='creativitips',
    version='0.0.1',
    author='Mattia Barbaresi',
    author_email='mattia.barbaresi@gmail.com',
    description='Segmentation, Chunking and Creative Generation with TPs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mattia-barbaresi',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
)
