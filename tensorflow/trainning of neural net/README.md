Dans cet implémetation du réseau de neuronne nous avons traiter le problème de l'explosion et de disparition du gradient.
Les causes de ces problèmes ont été en partie déterminé par Glorot et Xavier. Ils sont du à l'utilisation de la fonction
d'activation logistique et  d'initialiser les poids des couches avec des valeurs alèatoires.


L'utilisation de cette méthode entraine l'augmentation de la variance à la sortie de chaque noeud des neuronnes. Cette 
variance augmente couche après couche jusqu'à atteindre saturation. La fonction logistique étant une fonction qui pour une
grande valeur positive ou négative sature à 1 (positif) ou 0 (négatif) avec une dérivé proche de zéro. 

\frac{\mathrm{d}y}{\mathrm{d}x} = \frac{\lambda e^{-\lambda x}}{(1 + e^{-\lambda x})^2}

celle ci doit resté la même à la sortie lors de la propagation ainsi que lors de la rétropropagation.
