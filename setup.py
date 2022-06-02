import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open('requirements.txt') as fr:
    required = fr.read().splitlines()

setuptools.setup(
    name='creativitips',
    version='0.0.1',
    author='Mattia Barbaresi',
    author_email='mattia.barbaresi@gmail.com',
    description='Segmentation, Chunking and Creative Generation with TPs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mattia-barbaresi',
    # project_urls={
    #     "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    # },
    license='MIT',
    packages=['creativitips'],
    install_requires=['required'],
)
