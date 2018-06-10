import tensorflow as tf
from PIL import Image
import numpy as np

try:
    from models import *
except Exception:
    from .models import *

RANDOM_SEED_SIZE = 128

MNIST_IMG_W = 28
MNIST_IMG_H = 28

DISCRIMINATOR_SCOPE = 'discriminator'
GENERATOR_SCOPE = 'generator'

BATCH_SIZE = 100

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', './logs/', 'Directory where model is saved.')
tf.app.flags.DEFINE_string('out_dir', './outputs/', 'Directory where save generated images.')
tf.app.flags.DEFINE_integer('num_imgs', BATCH_SIZE, 'Number of images to generate.')


def load_saved_model(sess, saver):
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
    print('Loaded pre-trained model')


def main(args):
    int_sqrt_floor = int(np.floor(np.sqrt(FLAGS.num_imgs)))
    int_sqrt_ceil = int(np.ceil(np.sqrt(FLAGS.num_imgs)))

    random_seed = tf.random_uniform([FLAGS.num_imgs, RANDOM_SEED_SIZE], -1., 1.)

    with tf.variable_scope(GENERATOR_SCOPE):
        generator = get_generator(random_seed)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GENERATOR_SCOPE)
    saver = tf.train.Saver(var_list=g_vars, pad_step_number=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        load_saved_model(sess, saver)

        images = sess.run(generator).reshape(-1, MNIST_IMG_W, MNIST_IMG_H)

        result_img = Image.new('L', (int_sqrt_floor * MNIST_IMG_W, int_sqrt_ceil * MNIST_IMG_H))

        for i, img in enumerate(images):
            im = Image.fromarray((img * 255).astype('uint8'))
            result_img.paste(im, ((i % int_sqrt_floor) * MNIST_IMG_W, (i // int_sqrt_ceil) * MNIST_IMG_H))

        result_img.save('./{}/result.jpg'.format(FLAGS.out_dir))


if __name__ == '__main__':
    tf.app.run()
