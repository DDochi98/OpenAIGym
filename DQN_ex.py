import gym
from stable_baselines3 import DQN
import time 

def train(env_name, algo_class, model_path):
    env = gym.make(env_name)

    model = algo_class("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=10000)

    model.save(model_path)

def run(env_name, algo_class, model_path):
    env = gym.make(env_name)

    #Model load
    model = algo_class.load(model_path, env, verbose=1)
    
    #Run the model
    obs = env.reset()

    #Show reward
    rewards = 0

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        rewards += reward
        if done:
            time.sleep(0.5)
            print("n_steps : {}".format(rewards))
            rewards = 0
            obs = env.reset()

if __name__ == "__main__":
    env_name = "CartPole-v1"
    algo_class = DQN
    model_path = "model/model_cartpole"

    #train(env_name,algo_class,model_path)
    run(env_name,algo_class,model_path)