import gym
import numpy as np

cliffEnv = gym.make("CliffWalking-v0")

done = False
state = cliffEnv.reset()
while not done:
    print(cliffEnv.render(mode="ansi"))
    action = int(np.random.randint(0, 4, 1))
    print(state, "-->", ['up', 'right', 'down', 'left'][action])
    state, reward, done, _ = cliffEnv.step(action)

    print(state)
cliffEnv.close()
