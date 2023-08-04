import gym
import panda_gym

env = gym.make('PandaReach-v2', render=True)

obs = env.reset()

done = False

for _ in range(1000):
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    #env.render()

env.close()