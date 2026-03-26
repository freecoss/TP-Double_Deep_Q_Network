# TP 2 : Double Deep Q Network (Double DQN)

## Description de l'exercice

Ce projet implémente un agent **Double DQN** pour résoudre un problème de navigation dans un environnement GridWorld 4x4. L'objectif est d'atteindre la case cible (3,3) en partant de la case de départ (0,0) tout en évitant un obstacle en position (1,1).

## Problème résolu par Double DQN

Dans un DQN classique, les valeurs Q peuvent être surestimées car le même réseau est utilisé pour sélectionner ET évaluer les actions. Cela crée des problèmes de divergence et rend l'apprentissage instable.

**Double DQN** résout ce problème en séparant la sélection et l'évaluation :
- **Réseau principal (online)** : Sélectionne la meilleure action → a' = argmax Q_θ(s', a')
- **Réseau cible (target)** : Évalue cette action → Q_target(s,a) = r + γQ_θ'(s', a')

## Structure du code

### Fichiers principaux

- `double_dqn.py` : Implémentation complète de l'agent Double DQN et entraînement
- `test_double_dqn.py` : Script de test du modèle entraîné
- `double_dqn_model.keras` : Modèle entraîné sauvegardé

### Classes principales

#### GridWorld
Environnement de simulation 4x4 avec :
- Position de départ : (0, 0)
- Position cible : (3, 3)
- Obstacle : (1, 1)
- Récompenses : +10 (objectif), -5 (obstacle), -1 (déplacement normal)

#### DoubleDQNAgent
Agent d'apprentissage par renforcement avec :
- **Deux réseaux de neurones** : model (online) et target_model (target)
- **Mémoire de replay** : Stocke 2000 expériences
- **Stratégie ε-greedy** : Balance exploration/exploitation
- **Méthode replay()** : Implémente la logique Double DQN

### Paramètres d'entraînement

```python
GRID_SIZE = 4
GAMMA = 0.9              # Facteur de réduction
LEARNING_RATE = 0.01     # Taux d'apprentissage
EPSILON = 1.0            # Exploration initiale
EPSILON_DECAY = 0.995    # Décroissance de l'exploration
BATCH_SIZE = 32          # Taille du batch
MEMORY_SIZE = 2000       # Taille de la mémoire
EPISODES = 1000          # Nombre d'épisodes
TARGET_UPDATE_FREQ = 10  # Fréquence de mise à jour du réseau cible
```

## Architecture du réseau

```
Input Layer  : 16 neurones (état 4x4 aplati)
Hidden Layer : 24 neurones (ReLU)
Hidden Layer : 24 neurones (ReLU)
Output Layer : 4 neurones (Q-values pour chaque action)
```

## Résultats de l'entraînement

L'agent a été entraîné sur 1000 épisodes. Au cours de l'entraînement :
- L'epsilon décroît progressivement de 1.0 à ~0.01
- Le score moyen augmente au fil des épisodes
- L'agent apprend à éviter l'obstacle et à atteindre l'objectif

## Résultats des tests

Le modèle entraîné a été testé sur 10 épisodes :

**Taux de réussite** : 10/10 (100%)

### Performances observées

- **Score moyen** : 5 points par épisode
- **Nombre d'étapes** : 6 étapes (chemin optimal)
- **Chemin trouvé** : (0,0) → (0,1) → (0,2) → (0,3) → (1,3) → (2,3) → (3,3)

L'agent entraîné est capable de :
- Naviguer efficacement de (0,0) vers (3,3) avec un taux de réussite de 100%
- Éviter l'obstacle en (1,1) en contournant par le haut
- Trouver un chemin optimal en 6 étapes de manière consistante

## Utilisation

### Entraînement du modèle

```bash
.\tf_env\Scripts\python.exe double_dqn.py
```

### Test du modèle

```bash
.\tf_env\Scripts\python.exe test_double_dqn.py
```

## Avantages du Double DQN

1. **Évite la surestimation** des valeurs Q
2. **Stabilise l'apprentissage** et améliore la convergence
3. **Meilleures performances** que DQN classique sur des problèmes complexes

## Différence avec DQN classique

**DQN classique** :
```python
target[action] = reward + GAMMA * np.max(self.model.predict(next_state))
```

**Double DQN** :
```python
best_action = np.argmax(self.model.predict(next_state))  # Sélection
target[action] = reward + GAMMA * self.target_model.predict(next_state)[best_action]  # Évaluation
```
