During my experiments with GAN I learned that balance between Generator and Discriminator is crucial. With too _weak_ Discriminator, Generator wouldn't be able to learn. On the other hand, too _strong_ Discriminator learns very fast and makes impossible for Generator to train.

I have read lots of papers about this problem and it seems that improving and stabilizing the way in which GANs are trained is still open problem (03.09.2017).

I found really interesting concept called Wasserstein GAN [[1]](https://arxiv.org/abs/1701.07875), [[2]](https://arxiv.org/abs/1704.00028). The authors claim, that using Earth Movers metric during training of GANs allows them to train Generator and Critic (their name for Discriminator) without paying much attention to keeping perfect balance between networks.

The gradient penalty does indeed stabilize training and allows to use deeper architectures. I tested this method only on DCGAN-like network and MNIST dataset, but authors show images generated by for eg. ResNet on LSUN.

Results after 25000 steps with batch 4096:

![Results](https://github.com/mystic123/DeepLearning/blob/master/GAN/wGAN-MNIST/generated_imgs/img.jpg)