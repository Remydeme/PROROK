import tensorflow as tf
import numpy as np





# dans ce fichier nous allons construire un réseau de neuronne à laide de tensorflow
# et utiliser différente technique afin de régler les hyperparamètre.



"""
Dans cet implémetation du réseau de neuronne nous avons traiter le problème de l'explosion et de disparition du gradient.
Les causes de ces problèmes ont été en partie déterminé par Glorot et Xavier. Ils sont du à l'utilisation de la fonction
d'activation logistique et  d'initialiser les poids des couches avec des valeurs alèatoires.

L'utilisation de cette méthode entraine l'augmentation de la variance à la sortie de chaque noeud des neuronnes. Cette 
variance augmente couche après couche jusqu'à atteindre saturation. La fonction logistique étant une fonction qui pour une
grande valeur positive ou négative sature à 1 (positif) ou 0 (négatif) avec une dérivé proche de zéro. 

celle ci doit resté la même à la sortie lors de la propagation ainsi que lors de la rétropropagation. 

"""


# input : picture of 28 * 28 pixels in black and white
# This neural net is a perceptron neural net with 3 hidden
# activation functions: leaky relu
# optimizer : adam
class NN():

    input_size = 28 * 28
    hidden_1 = 300
    hidden_2 = 100
    hidden_3 = 50
    ouput = 10


    # scope : name of the scope use for all the layer
    # kernel_initializer : We init the weight using he technic.
    def build_net(self, inputs, scope):
        with tf.name_scope(scope)
            he_init = tf.contrib.layers.variance_scaling_initializer()
            X = tf.layers.dense(inputs,activation=tf.nn.leaky_relu, kernel_initializer=he_init, name='X')