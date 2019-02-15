from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Projekat iz softa na cloudu',
      author='Igor Majic',
      author_email='majic753@gmail.com',
      install_requires=[
          'keras',
          'h5py',
          'opencv-python',
          'numpy',
          'scikit-learn',
          'tensorflow'
      ],
      zip_safe=False)
