import io
import re
from setuptools import setup, find_packages

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name='mylib',
    version="v0.1",
    license='MIT',
    author='Molozhavenko Alexander',
    author_email="molozhavenko.aa@phystech.edu",
    description='mylib, python package',
    long_description=readme,
    url='https://github.com/intsystems/Molozhavenko-BS-Thesis',

    # options
    packages=find_packages(),
    install_requires=requirements,
)
