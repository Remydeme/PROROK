# Q-Learning

French:

### Epsilon Greedy l'exploration et l'exploitation

Nous souhaitons faire se déplacer une voiture dans un environnement qui lui est inconnue au commencement. La voiture est appelée  agent en apprentissage supervisé (RF). Cette voiture se déplace sur une map, l'objectif étant d'atteindre la maison. Si la voiture attein  sont objectif elle obtient une récompense. Si en chemin elle effectue une mauvaise action elle obtiendra une récompense négative. 

### Epsilon Greedy 

Lors de l'apprentissage supervisé, notre voiture avance en respectant une police (pi). Cette police consiste à choisir l'action qui maximise l'espérance pour un état futur S' sachant que l'on est dans l'état S. En suivant cette logique de déplacement nous devons normalement atteindre notre objectif et obtenir une récompense, la plus proche. Mais admettons qu'il y avait une récompense plus éloigné et plus importante notre intelligence artificielle en suivant uniquement cette logique ne l'atteindra jamais.

L'epsilon greedy est un paramètre qui varie entre 0 et 1. Sa valeur détermine le pourcentage d'exploration que l'on souhaite effectuer. 0.3 impliquent 30 % d'exploration et 70 % d'exploitation. Ce paramètre d'exploration va permettre à notre agent de découvrir de nouveaux états car il effectuera 30 % de ces mouvements aléatoirement. 


### L'équation de Bellman

Q(s,a)new = r +  gamma * Max(Q(s',a')))

r : reward.
gamma : valeur qui influence influence la valeur que l'on apporte au état futur.
Max(Q(s', a')) : on choisit l'état futur qui maximise la Q-function.

L'équation de Bellman nous permet de calculer la valeur que l'on accorde à un état s sachant que nous souhaitons effectuer l'action a. Elle utilise l'état futur pour calculer cette valeur et le coefficient gamma permet de définir l'importance que l'on accorde aux états futur. 1 signifierait que l'on accorde la même importance à toutes les cases. En général, cette valeur est égale à 0.9, mais elle peut être adapté en fonction de l'environnement. 


### Update function

Q(s,a)new = Q(s,a)old - l * \[ r + gamma * (Q(s',a') - Q(s,a)old)\]

s : state 
a : action 
l : learning rate

### Implemntation


Le programme est implmenter en python. Pour obtenir un agent capabe de se déplacer sur notre map il nosu faut construire notre Q-Table est la remplir des valeur Q(s, a). Nous avons implementer deux classe :

- Agent : qui représente la voiture.
- Map : qui correpond au terrain sur lequel la voiture se déplace.

English: 

# Q-Learning

English:

### Epsilon Greedy Exploration and Exploitation

We want to move a car in an environment that is unknown to him at the beginning. The car is called a supervised learning (RF) agent. This car moves on a map, the goal being to reach the house. If the car reaches its goal it gets a reward. If on the way she performs a bad action she will get a negative reward.

### Epsilon Greedy

During supervised learning, our car advances by respecting a policy (pi). This policy consists in choosing the best future state knowing that one is in the state S. By following this logic of displacement we should normally reach our objective and obtain a reward, the nearest one. But let's admit that there was a more distant and important reward our artificial intelligence by following only this logic will never reach it.

The greedy epsilon is a parameter that varies between 0 and 1. Its value determines the percentage of exploration that you want to perform. 0.3 involve 30% exploration and 70% exploitation. This exploration parameter will allow our agent to discover new states because he will perform 30% of these moves randomly.


### The Bellman equation

Q (s, a) new = r + gamma * Max (Q (s ', a')))

r: reward.
gamma: the value that influences the value that we bring to the future state.
Max (Q (s ', a')): we choose the future state that maximizes the Q-function.

Bellman's equation allows us to calculate the value we give to a state if we want to perform action a. It uses the future state to calculate this value, and the gamma coefficient is used to define the importance that is given to future states. 1 would mean that all the boxes are given the same importance. In general, this value is equal to 0.9, but it can be adapted according to the environment.


### Update function

Q (s, a) new = Q (s, a) old - l * \[r + gamma * (Q (s ', a') - Q (s, a) old) \]

s: state
a: action
l: learning rate

