# Deep Q Learning

## Loss function

### Li(s, a, s', r) = E[(r + gamma * Q(s',a') - Q(s, a))^2]

## Implementation

Le deep q-learning utilise un réseau neuronal pour prédire les actions de l'agent. Il prend en entrer 
l'état  et en sortie nous donne une valeur comprise entre 0 et 1 qui correspond a notre Qtarget.

#### Comment nous l'avons implémenté

Nous avons créer des tableau afin de stoquer:

- Les rawards
- Les states
- Les actions
- Les états futurs

Dans une boucle se répétant n fois nous avons à chaque itération :

1 - Déterminé l'action futur.
2 - Déterminé la reward. 
3 - Déterminé l'atats futur.
4 - Stocké la rawards, l'état , l'état futurs et l'action.Et ceci dans un tableau à un index choisi aléatoirement.
5 - A la fin de la boucle on entraine notre modèle.

#### 5 Entrainement du modèle

L'entrainement se déroule en 3 étapes:

##### 5.1 Transformer nos array en Tensor
##### 5.2 Calculer Qtarget pour l'état future à l'aide du réseau de neuronnes.
 
 * Qtarget(S', a') = r + gamma * Q(S', A')
 
 Nous avons déja  r la récompense et gamma qui est une constante. il nous faut calculer Q(S', A) avec notre RN puis calculer 
 Qtarget. 
 
##### 5.3 Constituer des batch afin d'entrainer notre modèle sur des batchs de taille N.

Entrainer sur des bacth d'une certaine taille amèliore le temps d'entrainement.

##### 5.4 entrainer le réseau de neuronnes

A cette étapes nous avons Qtarget et il nous faut juste Q. Pour rappel :

Li(s, a, r, s') = E[ (r + gamma * Q(S', A') - Q(S, A))^2]

Qtarget = r + gamma * Q(S', A')
Q(S, A) s'obtient avec le réseau de neuronne on lui donne l'état S et il nous retourne en sortie plusieur valeur et on choisi
la plus grande d'entre elle.

