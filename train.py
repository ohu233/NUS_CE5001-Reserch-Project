'''
改进奖励函数：高峰时间
现在只是以6.18为dataset 问题在于怎么处理每天的数据(groupby?)
算法优化问题 每秒推进导致计算冗杂
自适应学习率
多智能体
动作空间可以不进入泊位
状态空间最近车辆到达情况，泊位使用情况， 线路状态
bays中间部分也可以视为一个队列
通过坐标来确认车辆到达的相似度，而不是通过时间
'''
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from Env import BusEnv
# load dataset
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
check_env(env, warn=True)

# tensorboard --logdir=logs/
log_dir = "logs/"
model = PPO('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=log_dir)

model.learn(total_timesteps=2048 * iteration)
model.save("Model/PPO_model")