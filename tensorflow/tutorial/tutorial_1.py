import tensorflow as tf



# how to implement a simple addition in tensorflow

def add(x, y):
    tf_x = tf.constant(x, dtype=tf.float32, name='x')
    tf_y = tf.constant(y,dtype=tf.float32, name='y')


    z = tf_x + tf_y

    # now we have just created our graph we need to execute the graph

    with tf.Session() as sess:

        result = sess.run(z)

        print(result)




"""

Dans ce code on commmence par creer deux variables X, et Y 
puis on cree notre graphe 
on cree un gloabl variable initializer afin d'initiliser toutes nos variables 
on évalue notre graph avec la commende sess.run(z)
on stock la valeur dans result puis on l'affiche 
"""


def addVariable(x, y):
    """ Implementation of th add function using variable"""
    tf_x = tf.Variable(x, dtype=tf.float32, name='x')
    tf_y = tf.Variable(y, dtype=tf.float32, name='y')

    z = tf_x + tf_y

    # now we have just created our graph we need to execute the graph

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

       # we nee to init the variable with the value i prefer to use global initializer
       # tf_x.initializer.run()
       # tf_y.initializer.run()
        init.run()






























        result = sess.run(z)

        print(result)

if __name__ == "__main__":
    add(4, 5)
    addVariable(4, 5)
