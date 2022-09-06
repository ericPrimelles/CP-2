from email import policy
from time import sleep
import numpy as np

np.set_printoptions(2)

def MCControl(env, epsilon=0.950, episodes=10000, gamma=0.99, alpha=0.2):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode = []
    rewards = np.zeros(episodes)
    # Generate Episode
    s = env.reset()
    action = None
    for i in range(episodes):
        print(f'Episode {i}')
        while 1:
            if np.random.rand() <= epsilon and epsilon > 0.01:
                action = env.action_space.sample()
            
            else:
                action = np.argmax(Q[s])
            
            s_1, rwd, done, _ = env.step(action)
            episode.append((s, action, rwd))
            s = s_1
            rewards[i] += rwd
            if done or rwd == -100:
                
                G = 0
                
                for j in reversed(range(len(episode))):
                        o, a, r = episode[j]
                        G = gamma * G + r
                        if not (o,a) in [(e[0], e[1]) for e in episode[0 : j - 1]]:
                            
                            Q[o,a] += alpha * (G - Q[o, a])
                
                episode = []
                env.reset()
                break
        epsilon -= 0.001
    policy = np.argmax(Q, axis= 1)
    return Q, policy, rewards

import gym
import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    env = gym.make('CliffWalking-v0')
    env.reset()
    Q, policy, rwds = MCControl(env)
    print(Q)
    
    print(policy)
    s = env.reset()
    env.render()
    '''while 1: 
        
        action = policy[s]
        s, r, done, _ = env.step(action)
        env.render()
        print(r)
        if done:
            break
    '''
    env.close()
    
    plt.plot(rwds)
    plt.show()
    