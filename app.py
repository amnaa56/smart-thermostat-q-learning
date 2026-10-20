
import random
import numpy as np


class Room:
    def __init__(self):
        self.reset()

    def reset(self):
        self.temp, self.occupied = 30, 1
        return self.state()

    def state(self):
        return (self.temp - 20, self.occupied)

    def step(self, action):
        # People may enter or leave the room
        if random.random() < 0.1:
            self.occupied = 1 - self.occupied

        # AC ON cools the room
        if action == 1:
            self.temp -= 1
        elif random.random() < 0.5:
            self.temp += 1

        self.temp = max(20, min(34, self.temp))

        energy = action
        discomfort = abs(self.temp - 24) if self.occupied else 0
        reward = -(energy + 0.5 * discomfort)

        return self.state(), reward


def train(episodes=500, steps=200):
    Q = np.zeros((15, 2, 2))
    a_lr, gamma, eps = 0.1, 0.9, 1.0
    env = Room()

    for _ in range(episodes):
        s = env.reset()

        for _ in range(steps):
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = int(np.argmax(Q[s]))

            s2, r = env.step(a)

            Q[s][a] += a_lr * (
                r + gamma * np.max(Q[s2]) - Q[s][a]
            )

            s = s2

        eps = max(0.05, eps * 0.99)

    return Q


def evaluate(policy, steps=1000):
    env = Room()
    s = env.reset()
    total = 0

    for t in range(steps):
        a = policy(s, t)
        s, r = env.step(a)
        total += r

    return total / steps


Q = train()

rl_policy = lambda s, t: int(np.argmax(Q[s]))

normal_policy = lambda s, t: 1 if s[0] + 20 > 24 else 0

print("Avg reward RL   :", evaluate(rl_policy))
print("Avg reward Normal:", evaluate(normal_policy))
