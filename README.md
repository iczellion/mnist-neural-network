# ai-courses

Code repository for AI courses

# Develop locally

```shell
# On first time cloning, install dependencies
pip install -r requirements.txt

# Extract the MNIST tar archive dataset to ./tmp
mkdir -p ./.tmp && tar -xf ./dataset/mnist.tar.gz -C ./.tmp --strip-components=1
```


# Run application

```
# Start jupyter
jupyter lab

# Run application
python src/nn.py

# Run tests
pytest -s
```