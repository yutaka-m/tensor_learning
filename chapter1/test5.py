# coding: UTF-8
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf

var1 = tf.Variable(0)
const2 = tf.constant(3)

add_op = tf.add(var1, const2)
update_var1 = tf.assign(var1, add_op)

mul_op = tf.mul(add_op, update_var1)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print(sess.run([mul_op]))
  print(sess.run([mul_op, mul_op]))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print(sess.run([mul_op]))
  print(sess.run([mul_op, mul_op]))


