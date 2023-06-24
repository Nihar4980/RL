import gym
import pickle as pkl
import numpy as np
import cv2
import show_event

cliffEnv = gym.make('CliffWalking-v0')

q_table = pkl.load(open('q_learning_q_table.pkl', 'rb'))

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(0,1,1))
    return action

NUM_EPISODE = 5
for episode in range(NUM_EPISODE):
    done = False
    total_reward = 0
    episode_length = 0
    frame = show_event.initialize_frame()
    state = cliffEnv.reset()
    while not done:
        frame2 = show_event.put_agent(frame.copy(),state)
        cv2.imshow('Cliff Walking', frame2)
        cv2.waitKey(250)

        action = policy(state)
        state, reward, done, _ = cliffEnv.step(action)

        episode_length += 1
        total_reward += reward

    print(f'Episode : {episode} Episode_length : {episode_length} Total Reward : {total_reward}')
cliffEnv.close()
