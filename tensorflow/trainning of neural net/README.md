# Gradient Descent 

La descente de gradient permet d'ajuster les poids en fonction de l'erreur calculer en sortie du réseau de neuronne. 

### Comment mettre à jour les poids

Pour mettre à jour les poids il faut déterminer dans quel sens la variation de ces poids font varier notre érreur ? 

w <- w - learning_rate * (df/dw)  

df/dw doit être calculé pour chaque poids dans chaque couche avec la chain rule.

# Explosion et disparition du gradient

Dans cet implémentation d'un réseau de neuronne nous avons traité le problème de l'explosion et de disparition du gradient.
Les causes de ces problèmes ont été en partie déterminé par Glorot et Xavier. Ils sont du à l'utilisation de la fonction
d'activation logistique et  d'initialiser les poids des couches avec des valeurs alèatoires.


L'utilisation de cette méthode entraine l'augmentation de la variance à la sortie de chaque noeud des neuronnes. Cette 
variance augmente couche après couche jusqu'à atteindre saturation. La fonction logistique étant une fonction qui pour une
grande valeur positive ou négative sature à 1 (positif) ou 0 (négatif) avec une valeur de dérivé proche de zéro on observe une disparition du gradient lors de la rétropropagation. 

\frac{\mathrm{d}y}{\mathrm{d}x} = \frac{\lambda e^{-\lambda x}}{(1 + e^{-\lambda x})^2}
# put image of the sigmoid derivate

celle ci doit resté la même à la sortie lors de la propagation ainsi que lors de la rétropropagation.

### Solution proposé par Xavier et He

La solution proposé afin de régler ce problème est l'utilisation de la mthode de Xavier et He pour nitialiser les poids. Ceux ci, sont initialisées aléatoirement en fonction du nombre d'entrés et de sorties (fan in and fan out). L'ecart type pour une distribution normal centré en 0  doit avoir pour valeur :

# afficher la formule en image ici 
sigma = sqrt(2/input_size + output_size)technique améliore accélère l'apprentissage.

Note : Elle est appelé initialisation de He lorsque l'on utilise une fonction d'activation relu ou ces autres variantes. celle ci ne prend en compte que les entrées des couches. 

sigma = sqrt(2 / input_size)

#### Code 

He :
he = tf.contrib.layers.variance_scaling_initializer()

Xavier : 

Par défaut elle est utilisé par tensorflow.
Cette 

### Utilisation de fonction d'activation différente


La fonction d'activation sigmoide était utilisé car elle a un comportement proche de celui du cerveau. Toutefois, des performance meilleurs ont été observé avec la fonction Relu. La fonction Relu ne sature pas pour des valeurs proche de un et est rapide a calculer. 

Malheureusement, la fonction Relu cause un problème surnomé *"dying relu". En effet, la fonction relu retourne une valeur null lorque la valeur Z qui lui est fourni (Z = W*X + b) est négative. Cela entraine la mort de neuronne durant la phase d'entrainement. Ces neuronnes ont pour valeur de sortie 0. Il est peu probable que ce neurone reprenne vie car le gradient vaut toujours 0 lorque la sortie du neuronne est 0.

Il est possible d'utiliser une variante appelé, leaky Relu qui au lieu de retourné 0 pour des valeur négative retourne 
(alpha * z). Alpha correspond à la pente de Z. La fuite sera d'autant plus importante que alpha sera grand. D'après les recherche le réseau de neuronne est très performant avec un alpha égal à 0,2.


La fonction Leaky Elu : 

- leakyElu(z) = alpha * (exp(z) -1) si z < 0
- z                             si z >= 0

Elle donne de très bon résultat, meilleur que ceux de leakyRelu cependant la dérivé est plus longue a caculer. 

### Normalisation par mini-lots

Comme nous l'avons dit plus haut les entrès des couches ont une variance qui augmente et cela est du au paramètre des couches précédente fonction d'activation et la valeur des poids des couches précédentes. Sergey Ioffe et Christian Szegedy propose une technique qui propose de normaliser les entrées de chaque couches et de les décaler. Cette technique utilise un lot de valeur afin de calculer les moyennes et écart-type d'ou sont nom.

Cette technique est utilisé pour les réseaux de neuronnes profond et a fait ces preuves. Elle réduit énormément le problème du gradient. 
