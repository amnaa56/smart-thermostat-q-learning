import random
import numpy as np

class Room:
    def __init__(self):
        self.reset()

    def reset(self):
        self.temp = 30
        self.occupied = 1
        return self.state()

    def state(self):
        return (self.temp - 20, self.occupied)      # temp range 20..34

    def step(self, action):
        # random person comes or leaves
        if random.random() < 0.1:
            self.occupied = 1 - self.occupied

        # temperature change
        if action == 1:                              # AC on
            self.temp -= 1
        elif random.random() < 0.5:                  # AC off, room heats up
            self.temp += 1
        self.temp = max(20, min(34, self.temp))

        energy = 1.0 if action == 1 else 0.0
        discomfort = abs(self.temp - 24) if self.occupied else 0
        reward = -(energy + 0.5 * discomfort)
        return self.state(), reward

def train(episodes=500, steps=200):
    Q = np.zeros((15, 2, 2))
    lr, gamma, eps = 0.1, 0.9, 1.0
    env = Room()
    for _ in range(episodes):
        s = env.reset()
        for _ in range(steps):
            a = random.randint(0, 1) if random.random() < eps else int(np.argmax(Q[s]))
            s2, r = env.step(a)
            Q[s][a] += lr * (r + gamma * np.max(Q[s2]) - Q[s][a])
            s = s2
        eps = max(0.05, eps * 0.99)
    return Q

def evaluate(policy, steps=2000):
    env = Room()
    s, total = env.reset(), 0
    for _ in range(steps):
        s, r = env.step(policy(s))
        total += r
    return total / steps

Q = train()
rl_policy = lambda s: int(np.argmax(Q[s]))
normal_policy = lambda s: 1 if s[0] + 20 > 24 else 0   # cool whenever temp > 24

print("Avg reward RL    :", evaluate(rl_policy))
print("Avg reward Normal:", evaluate(normal_policy))
