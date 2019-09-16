import gym
import numpy as np
from random import uniform
import matplotlib.pyplot as plt

def q_learning(env, alpha = 0.9, eps = 0.05, gamma = 0.9, n_episodes = 1000, eval = False):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    penalties = np.zeros(n_episodes)
    rewards = np.zeros(n_episodes)
    timesteps = np.zeros(n_episodes)

    for i in range(n_episodes):
        state = env.reset()
        done = False
        n_penalties = 0
        total_reward = 0
        t = 0
        
        while not done:
            action = np.argmax(q[state]) if uniform(0, 1) < (1 - eps) else env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            q[state, action] += alpha * (reward + gamma * q[next_state].max(axis = 0) - q[state, action])
            state = next_state
            total_reward += reward
            t += 1

            if reward == -10:
                n_penalties += 1

        penalties[i] += n_penalties
        rewards[i] += total_reward
        timesteps[i] += t

    if eval:
        x = range(n_episodes)
        plt.figure()
        plt.subplot(311)
        plt.plot(x, penalties)
        plt.title("Penalties")

        plt.subplot(312)
        plt.plot(x, rewards)
        plt.title("Rewards")

        plt.subplot(313)
        plt.plot(x, timesteps)
        plt.title("Timesteps")
        plt.show()


    return q.argmax(axis = 1)

def q_learning_vfa(env, alpha = 0.9, eps = 0.1, 
        gamma = 1, n_episodes = 1000, eval = False):
    w = np.zeros((*env.observation_space.shape, env.action_space.n))
    timesteps = np.zeros(n_episodes)
    w_sum = []

    for i in range(n_episodes):
        state = env.reset()
        done = False
        t = 0
        #alpha = alpha / (1 + 0.05 * i)

        while not done:
            q_hat = state @ w
            action = (np.argmax(q_hat) 
                        if uniform(0, 1) < (1 - eps) 
                        else env.action_space.sample())
            next_state, reward, done, _ = env.step(action)
            w[:, action] -= alpha * (reward + gamma * (next_state @ w).max() - q_hat[action]) * state
            w_sum.append(w.sum())
            t += 1
            state = next_state

        timesteps[i] = t

    if eval:
        x = range(n_episodes)
        plt.plot(x, timesteps)
        plt.title("Timesteps")

    return w, w_sum

def render_episode(env, w):
    state = env.reset()

    done = False

    while not done:
        env.render()
        action = np.argmax(state @ w)
        state, reward, done, _ = env.step(action)

    env.close()


def run():
    env = gym.make('Taxi-v2')
    policy = q_learning(env, eval=True)
    n_episodes = 1000
    penalties = np.zeros(n_episodes)

    for i in range(n_episodes):
        state = env.reset()

        done = False

        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)

            if reward == -10:
                penalties[i] += 1

    print(f"Average penalties: {penalties.mean()}")
