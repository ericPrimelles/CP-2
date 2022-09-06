from ast import While
from time import sleep

import numpy as np



        

def MCQPrediction(env, n_actions, n_spaces,  gamma=0.99, episodes=1000): # Value prediction
    
    Q = np.zeros((n_spaces, n_actions))
    R = {(i , j): [] for i in range(n_spaces) for j in range(n_actions)}
    episode = []
    for i in range(episodes):
        # Play Episode
        s = env.reset()
        while True:
            
            action = env.action_space.sample()
            s_1, rwd, done, _ = env.step(action)
            episode.append((s, action, rwd))
            s = s_1
            
            if done or rwd == -100:
                
                G = 0 
                    

                for j in reversed(range(len(episode))):
                    o, a, r = episode[j]
                    G = gamma * G + r
                    if not (o,a) in [(e[0], e[1]) for e in episode[0 : j - 1]]:
                        R[(o, a)].append(G)
                        Q[o, a] = np.mean(R[(o, a)])                   
                   
                env.reset()
                episode =[]
                break
        print(f'Episode {i} complete')
    return Q

             

import gym

if __name__ == "__main__":
    
    np.random.seed(7)
    env = gym.make('CliffWalking')
    s = env.reset()
    action_space = env.action_space.n
    state_space = env.observation_space.n
    
    Q = MCQPrediction(env, action_space, state_space, episodes=10000)
    print(Q)
    