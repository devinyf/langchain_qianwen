from setuptools import setup, find_packages

setup(
    name='langchain_qianwen',
    version='0.1.17',
    author='Devin YF',
    author_email='dyfsquall@qq.com',
    description='通义千问 Qianwen Langchain adapter',
    install_requires=[
        'langchain>=0.1.0',
        'dashscope>=1.9.0',
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DevinYF/langchain_qianwen',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
