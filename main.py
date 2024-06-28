#!/usr/bin/env python3

import glob
import os
import sys

try:
    # Adjust the path as necessary for your CARLA installation
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import random
import math

# Local imports from other modules
from env import CarEnv
from ddpg_agent import Agent

random_seed = random.randint(1, 25)
plt.ion()

# Initialize the Carla environment
env = CarEnv()

# Assuming agent and environment classes are properly defined and compatible
state_size = 2  # Example size, adjust based on actual state dimensions
action_size = 3  # Throttle, brake, steering
num_agents = 1  # Number of cars (agents) in the environment

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)

def ddpg(n_episodes=2000, max_t=1000, print_every=10, learn_every=20, num_learn=10, goal_score=30):
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes+1):
        env.reset()  # Reset the Carla environment
        state, image = env.get_state()  # image
        scores = np.zeros(num_agents)
        agent.reset()

        start_time = time.time()

        for t in range(max_t):
            actions = agent.act(state, image) #image
            (next_state, next_image), reward, done, _ = env.step(actions) #next_image
            agent.step(state, actions, reward, next_state, done, image, next_image) #image next_image
            state = next_state
            image = next_image #next_image
            scores += reward

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()

            if done:
                break

        mean_score = np.mean(scores)
        total_scores_deque.append(mean_score)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores_deque)
        duration = time.time() - start_time

        print('\rEpisode {}\tTotal Average Score: {:.2f}\tMean: {:.2f}\tDuration: {:.2f}'
              .format(i_episode, total_average_score, mean_score, duration), end="")

        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tTotal Average Score: {:.2f}'.format(i_episode, total_average_score))

        if total_average_score >= goal_score and i_episode >= 100:
            print('Problem Solved after {} episodes!! Total Average score: {:.2f}'.format(i_episode, total_average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return total_scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('train_scores_plot.png')
plt.show()
