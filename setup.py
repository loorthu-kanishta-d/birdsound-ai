from setuptools import setup, find_packages

setup(
    name='birdsound-ai',
    version='1.0.0',
    description='AI system for analyzing and classifying bird sounds from audio files',
    author='Loorthu Kanishta D',
    author_email='your-email@example.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'mysql-connector-python',
        'werkzeug',
        'opencv-python',
        'matplotlib',
        'pandas',
        'numpy',
        'scipy',
        'librosa',
        'python_speech_features',
        'scikit-image',
        'seaborn'
    ],
    python_requires='>=3.7',
)
