# mnist-neural-network

This repository contains a neural network implementation from scratch to train on the MNIST dataset.

## Setup

To set up the project locally, follow these steps:

1. **Extract the MNIST dataset:**
    ```shell
    # Extract the MNIST tar archive dataset to ./tmp
    mkdir -p ./.tmp && tar -xf ./dataset/mnist.tar.gz -C ./.tmp --strip-components=1
    ```

2. **Install the required Python packages:**
    ```shell
    pip install -r requirements.txt
    ```

## Usage

```
# Run application
python src/nn.py

# Run unit tests
pytest -s
```