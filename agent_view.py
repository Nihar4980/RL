import gym
import numpy as np
import show_event
import cv2

cliffEnv = gym.make("CliffWalking-v0")

done = False
frame = show_event.initialize_frame()
state = cliffEnv.reset()
while not done:
    frame2 = show_event.put_agent(frame.copy(),state)
    cv2.imshow('Cliff Walking', frame2)
    cv2.waitKey(250)
    action = int(np.random.randint(0, 4, 1))
   # print(state, "-->", ['up', 'right', 'down', 'left'][action])
    state, reward, done, _ = cliffEnv.step(action)

   # print(state)
cliffEnv.close()