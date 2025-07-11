echo "Downloading MNIST..."
# The urls below are not longer valid
# wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
# wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz

# See : https://github.com/fgnt/mnist
wget "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
wget "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz


echo "Extracting MNIST..."
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz

python convert.py train-images-idx3-ubyte images.txt
python convert.py train-labels-idx1-ubyte labels.txt
python sort.py
