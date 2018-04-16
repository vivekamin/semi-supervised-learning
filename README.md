# semi-supervised-learning

Considered 1000 labeled datapoints from MNIST with 100 labeled samples from each digit. Used the rest of the training images as unlabeled data. Trained the sparse autoencoder with p=0.1 and extracted the hidden representations for these labeled samples to get a matrix with dimensions [D × m] = [200 × 1000]. 

1]
Train a softmax classifier with these feature representations to classify the digits. Combined the encoder and the softmax classifier to create a classifier with network dimensions [784, 200, 10]. 

2]
Created another fully connected network with same dimensions [784, 200, 10] and initialize it with random values.Trained it with 1000 labeled samples from MNIST. 

Using these two networks, computed the classification accuracies for the unlabeled data.To compare the performance between them, in this competition sparse-autoencoder network[1] won.
