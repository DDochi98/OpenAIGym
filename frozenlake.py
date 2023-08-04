import gym
import numpy as np

env_name = "FrozenLake-v1"

env = gym.make(env_name)

alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.001

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

num_episodes = 1000

for _ in range(num_episodes):
	state = env.reset()
	done = False
	while not done:
		random_number = np.random.random()
		if random_number < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state])
		next_state, reward, done, info = env.step(action)
		td_target = reward + gamma * np.max(Q[next_state])
		td_error = td_target - Q[state, action]
		Q[state, action] = Q[state, action] + alpha * td_error
		state = next_state
		env.render()
	epsilon = max(epsilon - epsilon_decay, 0) 
	env.close()

#pickle로 저장
