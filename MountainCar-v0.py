import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")

MAXSTATES = 10**2
GAMMA = 0.9
ALPHA = 0.01

def max_dict(d):
    max_v = float('-inf')
    max_key = None
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v

def create_bins():
    bins = np.zeros((2, 10))
    bins[0] = np.linspace(-1.2, 0.6, 10)
    bins[1] = np.linspace(-0.07, 0.07, 10)
    # bins[2] = np.linspace(-.418, .418, 10)
    # bins[3] = np.linspace(-5, 5, 10)
    return bins

def assign_bins(observation, bins):
    state = np.zeros(2)
    for i in range(2):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def get_state_as_string(state):
    state_string = "".join(str(int(e)) for e in state)
    return state_string

def get_all_states_as_strings():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(2))
    return states

def initialize_Q():
    Q = {}
    all_states = get_all_states_as_strings()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()
    done = False
    count = 0
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _ = env.step(act)

        total_reward += reward

        if done and count < 200:
            reward = 100

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, count

def play_many_games(bins, N=10000):
    Q = initialize_Q()
    length = []
    reward = []

    for i in range(N):
        eps = 1.0 / np.sqrt(i+1)

        episode_reward, episode_length = play_one_game(bins, Q, eps)

        if i % 100 == 0:
            print(i, "%4f" % eps, episode_reward)

        length.append(episode_length)
        reward.append(episode_length)
    return length, reward, Q

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = np.mean(total_rewards[max(0, i-100): i+1])
        if running_avg[i] < 110:
            print("Won at", i)
    # print(total_rewards)
    # print(running_avg)
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

bins = create_bins()
lengths, rewards, Q = play_many_games(bins, 10000)
plot_running_avg(rewards)

done = False
count = 0
observation = env.reset()
total_reward = 0
while not done:
    env.render()
    count += 1
    state = get_state_as_string(assign_bins(observation, bins))
    act = max_dict(Q[state])[0]
    observation, reward, done, _ = env.step(act)
    total_reward += reward

print(count, total_reward)
