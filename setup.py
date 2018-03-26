#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='Neu',
    version='1.3',
    description='中文自然语言处理工具包',
    long_description=open('README.rst').read(),
    autuor='wujinhang, huxiao',
    author_email='wujinhang0729@gmail.com',
    license='BSD License',
    # packages=find_packages(),
    url='https://github.com/Woooooody/Neu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='NLP,tokenizing,Chinese word segementation',
    packages=['Neu'],
    package_dir={'Neu':'Neu'},
    package_data={'Neu':['*.*','model/*','nn/*']}

)
