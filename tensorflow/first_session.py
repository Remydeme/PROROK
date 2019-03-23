import tensorflow as tf



def add(a, b):
    x = tf.Variable(a, name='a')
    y = tf.Variable(b, name='b')
    c = y + x
    with tf.Session():
        x.initializer.run()
        y.initializer.run()
        print(c.eval())




def addConstant(a, b):
    x = tf.constant(a)
    y = tf.constant(b)
    z = x + y
    with tf.Session():
        print(z.eval())




if __name__ == "__main__":
    add(34, 32)
    addConstant(234, 234)

