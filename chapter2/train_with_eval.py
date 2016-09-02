# coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/',
        "チェックポイントを保存するディレクトリ")
tf.app.flags.DEFINE_string('test_data', None, "テストデータのパス")
import numpy as np
import model as model
import time as time
import os as os

from reader import Cifar10Reader

def _loss(logits, label):
    labels = tf.cast(label, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def _train(total_loss, global_step):
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op

def _eval(sess, top_k_op, input_image, label_placeholder):
    if not FLAGS.test_data:
        return np.nan
    image_reader = Cifar10Reader(FLAGS.test_data)
    true_count = 0
    for index in range(10000):
        image = image_reader.read(index)
        predictions = sess.run([top_k_op],
                feed_dict={
                    input_image: image.byte_array,
                    label_placeholder: image.label
                })
        true_count += np.sum(predictions)
    image_reader.close()
    return (true_count / 10000)
    

filenames = [
        os.path.join(
            FLAGS.data_dir,'data_batch_%d.bin' % i) for i in range(1, 6)
        ]

def main(argv = None):
    global_step = tf.Variable(9, trainable=False)
    train_placeholder = tf.placeholder(tf.float32,
            shape=[32, 32, 3],
            name='input_image')
    label_placeholder = tf.placeholder(tf.int32, shape=[1], name='label')
    image_node = tf.expand_dims(train_placeholder, 0)
    logits = model.inference(image_node)
    total_loss = _loss(logits, label_placeholder)
    train_op = _train(total_loss, global_step)
    top_k_op = tf.nn.in_top_k(logits, label_placeholder, 1)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        total_duration = 0
        for epoch in range(1, FLAGS.epoch + 1):
            start_time = time.time()
            for file_index in range(5):
                print('Epoch %d: %s' % (epoch, filenames[file_index]))
                reader = Cifar10Reader(filenames[file_index])
                for index in range(10000):
                    image = reader.read(index)
                    _, loss_value = sess.run([train_op, total_loss],
                            feed_dict={
                                train_placeholder: image.byte_array,
                                label_placeholder: image.label
                                })
                    assert not np.isnan(loss_value), \
                            'Model dicerged with loss = NaN'

                reader.close()
            duration = time.time() - start_time
            total_duration += duration
            prediction = _eval(sess, top_k_op,
                train_placeholder, label_placeholder)
            print('epoch %d duration = %d sec, prediction = %.3f'
                    % (epoch, duration, prediction))
            tf.train.SummaryWriter(FLAGS.checkpoint_dir, sess.graph)
        print('Total duration = $d sec' % total_duration)
            
if __name__ == '__main__':
    tf.app.run()
