from stable_baselines3 import PPO
import pandas as pd
from Env import BusEnv

data = pd.read_excel('Data/preprocessed(simple).xlsx')
data = data[data['ActualArrival'].dt.date == pd.to_datetime('2024-06-18').date()]

# parameters
iteration = 2000
lr = 3e-4
alpha = 1
beta = 1
gamma = 1
theta = 1


env = BusEnv(data)

model = PPO.load("Model/PPO_model.zip")

obs, info = env.reset()
for i in range(20000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, _ = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()