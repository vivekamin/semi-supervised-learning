#!/bin/bash


mkdir -p ${PWD}/data/mnist

if ! [ -e ${PWD}/data/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P ${PWD}/data/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d ${PWD}/data/mnist/train-images-idx3-ubyte.gz

if ! [ -e ${PWD}/data/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P ${PWD}/data/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d ${PWD}/data/mnist/train-labels-idx1-ubyte.gz

if ! [ -e ${PWD}/data/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P ${PWD}/data/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d ${PWD}/data/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e ${PWD}/data/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P ${PWD}/data/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d ${PWD}/data/mnist/t10k-labels-idx1-ubyte.gz
