import datetime
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

try:
    from models import *
except Exception:
    from .models import *

RANDOM_SEED_SIZE = 128

MNIST_IMG_W = 28
MNIST_IMG_H = 28
MNIST_IMG_CHAN = 1

DISCRIMINATOR_SCOPE = 'discriminator'
GENERATOR_SCOPE = 'generator'

LEARNING_RATE = 0.0002

BATCH_SIZE = 64
MAX_STEPS = 20000
LOGDIR = './logs/'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', LOGDIR, 'Directory where to write event logs and checkpoints.')
tf.app.flags.DEFINE_integer('max_steps', MAX_STEPS, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size.')


def main(args):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    run_name = '{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    run_dir = os.path.join(FLAGS.log_dir, run_name)

    mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    mnist_batch = tf.placeholder(tf.float32, [None, MNIST_IMG_W * MNIST_IMG_H], 'mnist_batch')
    real_images = tf.reshape(mnist_batch, [-1, MNIST_IMG_W, MNIST_IMG_H, MNIST_IMG_CHAN])

    random_seed = tf.random_uniform([FLAGS.batch_size, RANDOM_SEED_SIZE], -1., 1.)

    with tf.variable_scope(GENERATOR_SCOPE):
        generator = get_generator(random_seed)

    tf.summary.image('generator', generator)

    with tf.variable_scope(DISCRIMINATOR_SCOPE):
        real_logits, real_cls = get_discriminator(real_images)
        fake_logits, fake_cls = get_discriminator(generator, True)

    real_labels = tf.ones_like(real_logits)
    fake_labels = tf.zeros_like(fake_logits)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=real_labels))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=fake_labels))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=real_labels))

    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DISCRIMINATOR_SCOPE)
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GENERATOR_SCOPE)

    d_optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, global_step=global_step, var_list=g_vars)

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
            }

            t0 = time.time()

            if step % 2 == 0:
                _, d_loss_val = sess.run([d_optim, d_loss], feed_dict=feed_dict)
            _, g_loss_val, summary = sess.run([g_optim, g_loss, summary_op], feed_dict=feed_dict)

            t = time.time() - t0

            examples_per_sec = FLAGS.batch_size / t

            if step % 10 == 0:
                summary_writer.add_summary(summary, global_step=step)
                format_str = '{} step: {} d_loss: {:8f} g_loss: {:8f} ({:2f} ex/s)'
                dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(format_str.format(dt, step, d_loss_val, g_loss_val, examples_per_sec))

            if step > 0 and (step + 1) % 1000 == 0 or step == FLAGS.max_steps - 1:
                checkpoint_path = os.path.join(run_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
