import setuptools

setuptools.setup(
    name="meshed_memory_transformer", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    packages=[_ for _ in setuptools.find_packages() if 'm2transformer' in _],
    python_requires='>=3.6',
)
