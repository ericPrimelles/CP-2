import numpy as np
import matplotlib.pyplot as plt

def MCPIConvergence (n_points):
    pi_exp = np.zeros(n_points)
    points = []
    for i in range(n_points):
        print(f'Iteration {i}')
        point = np.random.uniform(0, 1, 2)
        points.append(point)
        in_count = 0
        for p in points:
            if np.hypot(p[0], p[1]) <= 1:
                in_count += 1
        
        in_count = float(in_count)
        pi = in_count * 4 / len(points)
        pi_exp[i] = pi
    
    return pi_exp


if __name__ == '__main__':
    
    n = 5000
    pi = MCPIConvergence(n)
    np.random.seed(7)
    
    plt.figure(1)
    plt.title('MonteCarlo Pi Estimation Convergence')
    plt.plot(np.full(n, np.pi))
    plt.xlabel('n_points')
    plt.ylabel('Estimation') 
    plt.plot(pi)
    
    plt.figure(2)
    plt.title('Estimation Error')
    plt.xlabel('n_points')
    plt.ylabel('Error') 
    plt.plot(np.abs(np.pi - pi))
    plt.show()    