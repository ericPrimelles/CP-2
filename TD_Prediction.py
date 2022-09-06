import numpy as np
from MC_Control import MCControl

def TDPrediction(env, episodes=10000, gamma=0.99, alpha=0.2):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(episodes):
        s = env.reset()
        
        while True:
            action = env.action_space.sample()
            s_1, r, done, _ = env.step(action)
            
            Q[s, action] += alpha * (r + gamma * Q[s_1, env.action_space.sample()] - Q[s, action] )
            s = s_1
            if done or r == -100:
                env.reset()
                break
    return Q        
            
import gym

if __name__ == '__main__':
    
    env = gym.make('CliffWalking-v0')
    
    print('MC-Control')
    print('TD-Prediction')
    Q_td = TDPrediction(env)
    print(Q_td)
    
            