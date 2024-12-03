# Env
## grid world
- [ ] implementation
## line world
- [ ] implementation
## tic tac toe 
- [ ] implementation
## - [x] Farkle : 
- [x] add others combinaison
- [x] fix the 1 - 11 - 11 scoring
- [x] add feature to calculate the right amount for the network
- [x] select all dice ?
- [x] fix - rejouer quand tout les dé sont pris
- [x] rethink the step with action logic
- [ ] Handle farkle from first dice (with score)

# Algo : - [ ]
- [X] Random
- [X] TabularQLearning (quand possible)
- [X] DeepQLearning
- -- explorer avec adam ? 
- [X] DoubleDeepQLearning
- [X] DoubleDeepQLearningWithExperienceReplay
- [X] DoubleDeepQLearningWithPrioritizedExperienceReplay
- [X] REINFORCE
- [ ] REINFORCE with mean baseline
- [ ] REINFORCE with Baseline Learned by a Critic
- [X] PPO A2C style
- [ ] RandomRollout
- [ ] Monte Carlo Tree Search (UCT)
- [ ] Expert Apprentice
- [ ] Alpha Zero
- [ ] MuZero
- [ ] MuZero stochastique


Métriques à obtenir (attention métriques pour la policy obtenue, pas pour la policy en mode entrainement)
:
- Score moyen (pour chaque agent) au bout de 1000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 10 000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 100 000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 1 000 000 parties d'entrainement (si possible)
- Score moyen (pour chaque agent) au bout de XXX parties d'entrainement (si possible)
- Temps moyen mis pour exécuter un coup
Si la partie est de durée variable :
- Longueur moyenne (nombre de step) d'une partie au bout de 1000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 10 000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 100 000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 1 000 000 parties d'entrainement (si
possible)
- Longueur moyenne d'une partie au bout de XXX parties (si possible)