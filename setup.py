from setuptools import setup, find_packages

setup(
    name='anpr-license-plate-recognition',
    version='1.0.0',
    author='Mehul Mittal',
    author_email='mehul.mittal@example.com',
    description='Automatic Number Plate Recognition using YOLOv3 and EasyOCR',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mehulmittal/ANPR-License-Plate-Recognition',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
    install_requires=[
        'opencv-python>=4.5.0',
        'easyocr>=1.6.0',
        'numpy>=1.21.0',
        'pyyaml>=6.0',
        'matplotlib>=3.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
    },
    entry_points={
        'console_scripts': [
            'anpr=src.main:main',
        ],
    },
)
