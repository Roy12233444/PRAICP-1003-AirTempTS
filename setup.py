from setuptools import setup, find_packages

setup(
    name="airtemp_agents",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'joblib',
        'scikit-learn',
    ],
    python_requires='>=3.7',
)
