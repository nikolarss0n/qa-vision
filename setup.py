from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qa_vision",
    version="0.1.0",
    author="AI Team",
    author_email="ai@example.com",
    description="A tool for detecting UI alignment issues using computer vision and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qa-vision",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "peft>=0.4.0",
        "datasets>=2.10.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "llama-cpp-python>=0.2.0",
        "tqdm>=4.64.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "training": [
            "accelerate>=0.20.0",
            "bitsandbytes>=0.41.0",
            "einops>=0.6.0",
            "sentencepiece>=0.1.97",
            "huggingface-hub>=0.16.4",
            "rouge-score>=0.1.2",
            "wandb>=0.15.5",
            "jsonlines>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qa-vision-train=scripts.train_lora:main",
            "qa-vision-inference=scripts.inference:main",
            "qa-vision-analyze=scripts.analyze_ui:main",
            "qa-vision-prepare=scripts.prepare_dataset:main",
        ],
    },
)