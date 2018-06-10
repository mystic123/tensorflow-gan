import datetime
import os
import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from models import get_generator, get_discriminator

RANDOM_SEED_SIZE = 128

MNIST_IMG_W = 28
MNIST_IMG_H = 28
MNIST_IMG_CHAN = 1

DISCRIMINATOR_SCOPE = 'discriminator'
GENERATOR_SCOPE = 'generator'

LEARNING_RATE = 0.0001
BETA_1 = 0
BETA_2 = 0.9
LAMBDA = 10

DISCRIMINATOR_ITERS = 5

BATCH_SIZE = 64
MAX_STEPS = 200000
LOGDIR = './logs/'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', LOGDIR, 'Directory where to write event logs and checkpoints.')
tf.app.flags.DEFINE_integer('max_steps', MAX_STEPS, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size.')


def main(args):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(FLAGS.log_dir, run_name)

    mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    training_phase = tf.placeholder_with_default(False, [], 'training_phase')

    mnist_batch = tf.placeholder(tf.float32, [None, MNIST_IMG_W * MNIST_IMG_H], 'mnist_batch')
    real_images = tf.reshape(mnist_batch, [-1, MNIST_IMG_W, MNIST_IMG_H, MNIST_IMG_CHAN])

    random_seed = tf.random_uniform([FLAGS.batch_size, RANDOM_SEED_SIZE], -1., 1.)

    with tf.variable_scope(GENERATOR_SCOPE):
        generator = get_generator(random_seed, training_phase)

    epsilon = tf.random_uniform(shape=(FLAGS.batch_size, 1, 1, 1), minval=0., maxval=1.)
    x_hat = epsilon * real_images + (1.0 - epsilon) * generator

    tf.summary.image('generator', generator)

    with tf.variable_scope(DISCRIMINATOR_SCOPE):
        real_logits, _ = get_discriminator(real_images, training_phase)
        fake_logits, _ = get_discriminator(generator, training_phase, True)
        rand_logits, _ = get_discriminator(x_hat, training_phase, True)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DISCRIMINATOR_SCOPE)
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GENERATOR_SCOPE)

    # gradient penalty
    grads_rand = tf.gradients(rand_logits, [x_hat])
    gradient_penalty = LAMBDA * tf.square(tf.norm(grads_rand[0], ord=2) - 1.0)

    # calculate discriminator's loss
    # d_loss = em_loss(real_labels, fake_logits) - em_loss(real_labels, rand_logits) + _gradient_penalty
    d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + gradient_penalty
    g_loss = -tf.reduce_mean(fake_logits)

    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)

    d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1, beta2=BETA_2).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1, beta2=BETA_2).minimize(g_loss,
                                                                                         global_step=global_step,
                                                                                         var_list=g_vars)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(var_list=tf.trainable_variables(), pad_step_number=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(run_dir, graph=sess.graph)

        for step in range(FLAGS.max_steps):
            feed_dict = {
                mnist_batch: mnist.train.next_batch(FLAGS.batch_size)[0],
                training_phase: True
            }

            t0 = time.time()

            _, d_loss_val = sess.run([d_optim, d_loss], feed_dict=feed_dict)

            if step > 0 and step % DISCRIMINATOR_ITERS == 0:
                _, g_loss_val, summary = sess.run([g_optim, g_loss, summary_op], feed_dict=feed_dict)

            t = time.time() - t0

            examples_per_sec = FLAGS.batch_size / t

            if step > 0 and step % 10 == 0:
                summary_writer.add_summary(summary, global_step=step)
                format_str = '{} step: {} d_loss: {:8f} g_loss: {:8f} ({:2f} ex/s)'
                dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(format_str.format(dt, step, d_loss_val, g_loss_val, examples_per_sec))

            if step > 0 and (step + 1) % 1000 == 0 or step == FLAGS.max_steps - 1:
                checkpoint_path = os.path.join(run_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
