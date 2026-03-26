import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4

# Déplacements possibles (Haut, Bas, Gauche, Droite) 
MOVES = { 
0: (-1, 0),  # Haut 
1: (1, 0),   # Bas 
2: (0, -1),  # Gauche 
3: (0, 1)    # Droite 
}

class GridWorld: 
    """Environnement GridWorld 4x4""" 
    def __init__(self): 
        self.grid_size = GRID_SIZE 
        self.reset() 
 
    def reset(self): 
        """Réinitialise l'agent à la position de départ.""" 
        self.agent_pos = (0, 0) 
        self.goal_pos = (3, 3) 
        self.obstacle_pos = (1, 1)
        return self.get_state() 
 
    def get_state(self): 
        """Retourne l'état sous forme d'un vecteur binaire.""" 
        state = np.zeros((GRID_SIZE, GRID_SIZE)) 
        state[self.agent_pos] = 1 
        return state.flatten() 
 
    def step(self, action): 
        """Fait avancer l'agent et renvoie (nouvel état, récompense, terminé).""" 
        x, y = self.agent_pos 
        dx, dy = MOVES[action] 
        new_x, new_y = x + dx, y + dy 
 
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE: 
            self.agent_pos = (new_x, new_y) 
 
        if self.agent_pos == self.goal_pos: 
            return self.get_state(), 10, True
        elif self.agent_pos == self.obstacle_pos: 
            return self.get_state(), -5, False
        else: 
            return self.get_state(), -1, False

# Charger le modèle entraîné
model = load_model("double_dqn_model.keras")
print("Modèle chargé avec succès!")

# Tester l'agent
env = GridWorld()
num_tests = 10

print(f"\n=== Test de l'agent sur {num_tests} épisodes ===\n")

total_success = 0
for test in range(num_tests):
    state = env.reset()
    total_reward = 0
    steps = 0
    path = [(0, 0)]
    
    for step in range(50):
        q_values = model.predict(np.array([state]), verbose=0)
        action = np.argmax(q_values[0])
        next_state, reward, done = env.step(action)
        
        path.append(env.agent_pos)
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            total_success += 1
            break
    
    print(f"Test {test+1}: Score = {total_reward}, Steps = {steps}, Succès = {done}")
    print(f"  Chemin: {' -> '.join([str(p) for p in path])}")

print(f"\n=== Résultats ===")
print(f"Taux de réussite: {total_success}/{num_tests} ({100*total_success/num_tests:.1f}%)")
