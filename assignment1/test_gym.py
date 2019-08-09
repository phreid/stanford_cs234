import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
env.close()