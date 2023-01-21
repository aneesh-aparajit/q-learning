import gym
import numpy as np
import warnings
warnings.simplefilter('ignore')

from tqdm import tqdm
from matplotlib import pyplot as plt
from model import Transforms, BreakoutModel
import torch as T
from PIL import Image
import os

transforms = Transforms()
model = BreakoutModel()
optimizer = T.optim.Adam(model.parameters(), lr=3e-5)

# LEARNING_RATE = 0.1
# DISCOUNT = 0.95
EPISODES = 2500
SHOW_EVERY =  500

LOW, HIGH = T.tensor(np.array([-1, -1]), requires_grad=False), T.tensor(np.array([1, 1]), requires_grad=False)
'''
The LOW and the HIGH values are -1 and 1 respectively because of the fact that, we use a Tanh activation in the neural network which compresses the output to the range -1 and 1.
'''


def train_q_model(WIN_SIZE, LEARNING_RATE, DISCOUNT):
    os.mkdir(f'../q_tables/{WIN_SIZE}_{LEARNING_RATE}_{DISCOUNT}/')

    print(f'WIN_SIZE: {WIN_SIZE} | LEARNING_RATE: {LEARNING_RATE} | DISCOUNT: {DISCOUNT}')

    env = gym.make("ALE/Breakout-v5")

    DISCRETE_OBS_SIZE = T.tensor([WIN_SIZE] * len(HIGH))
    DISCRETE_OBS_WINDOW_SIZE = (HIGH - LOW) / DISCRETE_OBS_SIZE

    ORIG_EPS = EPS = 0.5
    START_EPS_DECAY = 1
    END_EPS_DECAY = EPISODES // 2
    EPS_DECAY_VALUE = EPS / (END_EPS_DECAY - START_EPS_DECAY)

    Q_TABLE = T.tensor(np.random.uniform(low=-2, high=0, size=(list(DISCRETE_OBS_SIZE.cpu().detach().numpy()) + [env.action_space.n])))

    # print(env.action_space.n)

    ep_rewards = []
    aggr_ep_rewards = {
        'ep': [], 
        'avg': [], 
        'min': [], 
        'max': [], 
        'EPS': EPS, 
        'WIN_SIZE': WIN_SIZE,
        'LEARNING_RATE': LEARNING_RATE, 
        'DISCOUNT': DISCOUNT
    }

    BEST_SCORE = -1

    def get_discrete_states(state):
        discrete_state = (state - LOW) / DISCRETE_OBS_WINDOW_SIZE
        return tuple(discrete_state.type(T.int8))


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
                action = T.argmax(Q_TABLE[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            # action = np.argmax(Q_TABLE[discrete_state])

            new_state, reward, done, truncated,  _ = env.step(action=action)
            episode_reward += reward

            # print(episode_reward.requires_gra)

            # print(reward)

            new_state = Image.fromarray(np.uint8(new_state))
            new_state = transforms(new_state)
            new_state = model.forward(new_state)
            # new_state = new_state.cpu().detach().numpy()
            new_discrete_state = get_discrete_states(new_state)
            
            # if render:
            # env.render()

            if not done:
                # print(Q_TABLE[new_discrete_state])
                # print(T.max(Q_TABLE[new_discrete_state], dim=-1))
                max_future_q, _ = T.max(Q_TABLE[new_discrete_state], dim=-1)
                current_q = Q_TABLE[discrete_state + (action, )]
                new_q = current_q + LEARNING_RATE * ( reward + DISCOUNT * max_future_q - current_q)
                Q_TABLE[discrete_state + (action, )] = new_q
            
            # elif new_state[0] >= env.goal_position:
            #     print(f'\n\nCompleted at EPISODE {episode}!!!!!')
            #     Q_TABLE[discrete_state + (action, )] = 0

            # print(new_state)
            
            discrete_state = new_discrete_state

        if END_EPS_DECAY >= episode >= START_EPS_DECAY:
            EPS -= EPS_DECAY_VALUE

        # print(f'episode_reward: {episode_reward}')
        
        ep_rewards.append(episode_reward)
        # if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        if not episode % SHOW_EVERY:
            print(f"Episode: {episode}\t| ep_reward: {episode_reward:3f}\t| avg: {average_reward:3f}\t| min: {min(ep_rewards[-SHOW_EVERY:]):.3f}\t| max: {max(ep_rewards[-SHOW_EVERY:]):.3f}")

        # plt.figure()
        # plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
        # plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
        # plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
        # plt.legend(loc=4)
        # plt.show()
        # print(type(episode_reward), type(BEST_SCORE))
        if episode_reward  > BEST_SCORE:
            T.save(Q_TABLE, f'../q_tables/{WIN_SIZE}_{LEARNING_RATE}_{DISCOUNT}/q_table_{episode_reward}.pth')
            BEST_SCORE = episode_reward
    # import pickle
    # with open('../aggr.pickle', 'wb') as handle:
    #     pickle.dump(aggr_ep_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    env.close()

    aggr_ep_rewards['ep_rewards'] = ep_rewards

    return aggr_ep_rewards


if __name__ == '__main__':
    # EPSILON = [0.5, 0.6]
    WIN_SIZES = [10, 20, 30, 40, 50, 60]
    LEARNING_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
    DISCOUNTS = [0.99, 0.95, 0.9, 0.85, 0.8]


    results = {}

    # for EPS in EPSILON:
    for WIN_SIZE in WIN_SIZES:
        for LR in LEARNING_RATE:
            for DISCOUNT in DISCOUNTS:
                aggr_ep_rewards = train_q_model(WIN_SIZE=WIN_SIZE,  LEARNING_RATE=LR, DISCOUNT=DISCOUNT)
                results[(WIN_SIZE, LR, DISCOUNT)] = aggr_ep_rewards

    import pickle
    with open('../results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
