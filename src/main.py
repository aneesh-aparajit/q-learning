import gym
import numpy as np
import warnings
warnings.simplefilter('ignore')

from tqdm import tqdm
from matplotlib import pyplot as plt
from model import Transforms, BreakoutModel
import torch as T
from PIL import Image

transforms = Transforms()
model = BreakoutModel()


env = gym.make("ALE/Breakout-v5", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY =  500

LOW, HIGH = np.array([-1, -1]), np.array([1, 1])
'''
The LOW and the HIGH values are -1 and 1 respectively because of the fact that, we use a Tanh activation in the neural network which compresses the output to the range -1 and 1.
'''

DISCRETE_OBS_SIZE = [50] * len(HIGH)
DISCRETE_OBS_WINDOW_SIZE = (HIGH - LOW) / DISCRETE_OBS_SIZE

EPS = 0.5
START_EPS_DECAY = 1
END_EPS_DECAY = EPISODES // 2
EPS_DECAY_VALUE = EPS / (END_EPS_DECAY - START_EPS_DECAY)

print(f'discrete_obs_win_size: {DISCRETE_OBS_WINDOW_SIZE}')

Q_TABLE = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

print(env.action_space.n)

ep_rewards = []
aggr_ep_rewards = {
    'ep': [], 
    'evg': [], 
    'min': [], 
    'max': []
}

def get_discrete_states(state):
    discrete_state = (state - LOW)
    return tuple(discrete_state.astype(int))

for episode in tqdm(range(EPISODES)):
    state, info =  env.reset()
    state = Image.fromarray(np.uint8(state))
    state = transforms(state)
    state = model.forward(state)
    # state = state.cpu().detach().numpy()
    discrete_state = get_discrete_states(state)

    episode_reward = 0

    done = False
    # if episode % SHOW_EVERY == 0:
    #     render = True
    # else:
    #     render = False
    while not done:

        if np.random.random() > EPS:
            action = np.argmax(Q_TABLE[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        # action = np.argmax(Q_TABLE[discrete_state])

        new_state, reward, truncated, done, _ = env.step(action=action)
        episode_reward += reward

        new_state = Image.fromarray(np.uint8(new_state))
        new_state = transforms(new_state)
        new_state = model.forward(new_state)
        # new_state = new_state.cpu().detach().numpy()
        new_discrete_state = get_discrete_states(new_state)
        
        # if render:
        env.render()

        if not done:
            max_future_q = np.max(Q_TABLE[new_discrete_state])
            current_q = Q_TABLE[discrete_state + (action, )]
            new_q = current_q + LEARNING_RATE * ( reward + DISCOUNT * max_future_q - current_q)
            Q_TABLE[discrete_state + (action, )] = new_q
        
        elif new_state[0] >= env.goal_position:
            print(f'\n\nCompleted at EPISODE {episode}!!!!!')
            Q_TABLE[discrete_state + (action, )] = 0

        # print(new_state)
        
        discrete_state = new_discrete_state

    if END_EPS_DECAY >= episode >= START_EPS_DECAY:
        EPS -= EPS_DECAY_VALUE
    
    ep_rewards.append(episode_reward)
    # if not episode % SHOW_EVERY:
    average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
    aggr_ep_rewards['ep'].append(episode)
    aggr_ep_rewards['avg'].append(average_reward)
    aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
    aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

    # if not episode % SHOW_EVERY:
    print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

    plt.figure()
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
    plt.legend(loc=4)
    # plt.show()

env.close()
