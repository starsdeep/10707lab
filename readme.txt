Author: Yikang Liao
Andrew id: yliao1


1. Introduction 

I use the framwork provides by cs231n assignment http://cs231n.github.io/assignments2017/assignment2/.
I only use the provided code skeleton and implement all key components myself, for example,  forword, backword of affine layers, softmax layer, RBM models and Autoencoder model and denoising autoencoder model.

The implementation of RBM model is in 231n/classifiers/rbm.py
The implementation of Autoencoder and Denoising Autoencoder model is in 231n/classifiers/autoencoder.py
The experiments code is in RBM.ipynb

2. How to Run

2.1 Setup
cd assignment2
sudo pip install virtualenv      # This may already be installed
python3 -m venv .env             # Create a virtual environment (python3)
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies

Download train, validation and test data from http://www.cs.cmu.edu/~rsalakhu/10707/assignments.html to the root directory and named them as digitstrain.txt, digitsvalid.txt and digitstest.txt  

2.2 Run Experiments

jupyter notebook RBM.ipynb













