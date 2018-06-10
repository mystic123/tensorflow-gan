MNIST digits generator, my code is based mostly on Ian Goodfellow's [paper](https://arxiv.org/abs/1406.2661) and many tutorials/examples found online (blogs/github).

Results after 5000 steps with batch 512 (~50 epochs):
![Results](https://github.com/mystic123/tensorflow-gan/raw/master/GAN-MNIST/generated_imgs/img.jpg)

Adding Batch normalization layers to the Generator accelerates training, but major drawback of using BN layers is that it significantly increases computation time.
