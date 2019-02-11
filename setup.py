from setuptools import setup, find_packages

setup(
    name='Natrix',
    version='0.0.1.alpha',
    packages=find_packages(exclude='demo'),
    url='https://github.com/fbertola/Natrix',
    license='MIT',
    author='Federico Bertola',
    author_email='fb@bendingspoons.com',
    description='',
    install_requires=[
        "moderngl",
        "numpy",
        "jinja2",
        "python-decouple",
        "pyglet"
    ],
)
