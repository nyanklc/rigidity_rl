import gymnasium as gym

space = gym.spaces.Discrete(5)

for _ in range(20):
    print(space.sample())