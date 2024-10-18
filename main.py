import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class BusEnv(gym.Env):
    def __init__(self, data):
        super(BusEnv, self).__init__()
        '''
        capacity: capacity of each bay
        bay_num: number of bay
        service_time: time of bus stopping in bays
        '''
        self.capacity = 2
        self.bay_num = 2
        self.service_time = 60
        '''
        action space:
        0->bay1, 1->bay2, 2->wait in bay1 queue, 3->wait in bay2 queue
        '''
        self.action_space = spaces.Discrete(4)
        '''
        observation_space:
        bay1_1, bay1_2, bay2_1, bay2_2, ServiceNo, Load, Type, DayOfWeek, Latitude, Longitude
        '''
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 
            high=np.array([1, 1, 1, 1, 17, 3, 2, 7, 1.4 ,104], dtype=np.float32), 
            dtype=np.float32
        )
        '''
        load data(excel)
        change variable format
        '''
        self.data = data
        self.data['ActualArrival'] = pd.to_datetime(self.data['ActualArrival'])
        '''
        set start time
        per second
        '''
        self.current_time = self.data['ActualArrival'].min()
        self.time_step = pd.Timedelta(seconds=1)
        self.current_step = 0
        self.waiting_queue = {0: [], 1: []} 

        self.reset()

    def reset(self, seed=None, options=None):
        '''
        set random seed
        set bays status void
        '''
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.bays = [[None for _ in range(self.capacity)] for _ in range(self.bay_num)]
        self.remaining_service_time = [[0 for _ in range(self.capacity)] for _ in range(self.bay_num)]
        # set start time
        self.current_time = self.data['ActualArrival'].min()
        self.current_step = 0
        self.waiting_queue = {0: [], 1: []} 

        return self._next_observation(),{}

    def _next_observation(self):
        '''
        get bays info
        '''
        bay_status = np.array([1 if self.bays[i][j] is not None else 0 for i in range(self.bay_num) for j in range(self.capacity)], dtype=np.float32)

        #if bus arrivals
        if self.current_step < len(self.data) and self.data.iloc[self.current_step]['ActualArrival'] <= self.current_time:
            next_vehicle = self.data.iloc[self.current_step]
            vehicle_info = np.array([next_vehicle['ServiceNo'], 
                                     next_vehicle['Load'], 
                                     next_vehicle['Type'],
                                     next_vehicle['DayOfWeek'],
                                     next_vehicle['Latitude'],
                                     next_vehicle['Longitude']],
                                     dtype=np.float32)
        else:
            vehicle_info = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        return np.concatenate([bay_status, vehicle_info])

    def reward(self):
        '''
        calculate reward
        '''
        reward = 0
        # 奖励进入泊位的车辆
        for i in range(self.bay_num):
            for j in range(self.capacity):
                if self.bays[i][j] is not None:
                    service_no = self.bays[i][j]
                    load = self.data.loc[self.data['ServiceNo'] == service_no, 'Load'].values[0]
                    type = self.data.loc[self.data['ServiceNo'] == service_no, 'Type'].values[0]
                    dayofweek = self.data.loc[self.data['ServiceNo'] == service_no, 'DayOfWeek'].values[0]

                    reward += load * 2
                    if type == 2:
                        reward += 20

        for i in range(self.bay_num):
            reward -= len(self.waiting_queue[i]) * 3

        return float(reward)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        '''
        check arrival
        '''
        if self.current_step < len(self.data) and self.data.iloc[self.current_step]['ActualArrival'] <= self.current_time:
            current_vehicle = self.data.iloc[self.current_step]
            self.current_step += 1

            if action == 0 or action == 1:
                # Assign to Bay1 or Bay2
                if None in self.bays[int(action)]:
                    empty_spot = self.bays[int(action)].index(None)
                    self.bays[int(action)][empty_spot] = current_vehicle['ServiceNo']
                    self.remaining_service_time[int(action)][empty_spot] = self.service_time
                    reward = self.reward()  # if get in bays
                else:
                    # if full, assign in waiting queue
                    self.waiting_queue[int(action)].append(current_vehicle.to_dict())
                    reward = self.reward()  # if waiting
            elif action == 2 or action == 3:
                # Wait in Bay1 or Bay2 queue
                self.waiting_queue[int(action) - 2].append(current_vehicle.to_dict())
                reward = self.reward()  # if waiting

        # reduce service time and check bus status in waiting queue
        for i in range(self.bay_num):
            for j in range(self.capacity):
                if self.remaining_service_time[i][j] > 0:
                    self.remaining_service_time[i][j] -= 1
                if self.remaining_service_time[i][j] == 0 and self.bays[i][j] is not None:
                    # if waiting queue
                    self.bays[i][j] = None  # release bays 
                    if len(self.waiting_queue[i]) > 0:
                        next_in_queue = self.waiting_queue[i].pop(0)  # pop first bus in waiting queue
                        self.bays[i][j] = next_in_queue['ServiceNo']
                        self.remaining_service_time[i][j] = self.service_time  # reassign bus
                        #print(f"Vehicle {next_in_queue['ServiceNo']} from queue assigned to Bay {i+1}")

        # time + 1 second
        self.current_time += self.time_step

        # all buses assigned
        if self.current_step >= len(self.data) and all(len(queue) == 0 for queue in self.waiting_queue.values()):
            terminated = True  # simulation naturally terminates

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"current time: {self.current_time}")
        print(f"bay status: {self.bays}")
        print(f"remaining time: {self.remaining_service_time}")
        print("waiting queue:")
        for bay, buses in self.waiting_queue.items():
            print(f" Bay {bay + 1}:")
            for bus in buses:
                print(f"  ServiceNo: {bus['ServiceNo']}, ActualArrival: {bus['ActualArrival']}")
        print('-' * 40)

class BusSimularity():
    def __init__(self):
        pass


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