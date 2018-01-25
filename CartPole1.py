import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")

MAXSTATES = 10**4
GAMMA = 0.09
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
    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)
    return bins

val = [[0 for _ in range(10)] for _ in range(4)]

def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
        if state[i] == 10:
            state[i] = 9
        val[i][int(state[i])] += 1
    return state

def get_state_as_string(state):
    state_string = "".join(str(int(e)) for e in state)
    if len(state_string) == 5:
        print(state_string, state)
    return state_string

def get_all_states_as_strings():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
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

        if done:
            reward = -500

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
        if running_avg[i] > 475:
            print("Won at", i)
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

bins = create_bins()
lengths, rewards, Q = play_many_games(bins, 10000)
plot_running_avg(rewards)

print(val)

# states array:
# [[0, 0, 0, 933420, 1537631, 1323755, 207603, 58394, 0, 0], [0, 0, 27, 24399, 873057, 2112588, 811450, 201268, 34747, 3267], [0, 0, 817, 44469, 746681, 1787211, 1396592, 83900, 1133, 0], [0, 0, 955, 10528, 776014, 2464840, 780574, 26941, 951, 0]]


done = False
count = 0
observation = env.reset()

while not done:
    env.render()
    count += 1
    state = get_state_as_string(assign_bins(observation, bins))
    act = max_dict(Q[state])[0]
    observation, reward, done, _ = env.step(act)

print(count)
