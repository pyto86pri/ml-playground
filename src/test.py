import tensorflow as tf
print(tf.__version__)

x = tf.Variable(3, name='x')
y = tf.Variable(3, name='y')

f = x*x*y + y + 2

print(f.numpy())

