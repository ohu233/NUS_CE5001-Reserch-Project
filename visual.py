import pygame
import pandas as pd
from Env import BusEnv
from stable_baselines3 import PPO

class PygameVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen_width = 400
        self.screen_height = 800 
        self.bay_width = 150
        self.bay_height = 100
        self.queue_width = self.bay_width
        self.queue_height = 40 
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Bus Bay Visualization')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def update(self):
        self.screen.fill((255, 255, 255))

        for i in range(self.env.bay_num):
            for j in range(self.env.capacity):
                x = i * (self.bay_width + 10) + 50
                y = (self.screen_height // 2 - self.bay_height // 2) - 250 + j * (self.bay_height + 10)
                color = (0, 255, 0) if self.env.bays[i][j] is None else (255, 0, 0)
                pygame.draw.rect(self.screen, color, (x, y, self.bay_width, self.bay_height))

                if self.env.bays[i][j] is not None:
                    service_no = str(self.env.bays[i][j])

                    arrival_time = self.env.data.loc[self.env.data['ServiceNo'] == self.env.bays[i][j], 'ActualArrival'].values[0]
                    arrival_time_str = pd.to_datetime(arrival_time).strftime('%H:%M:%S')

                    service_text = f"No: {service_no}"
                    arrival_text = f"Arr: {arrival_time_str}"
                    text_surface_service = self.font.render(service_text, True, (0, 0, 0))
                    text_surface_arrival = self.font.render(arrival_text, True, (0, 0, 0))

                    self.screen.blit(text_surface_service, (x + 10, y + 10))
                    self.screen.blit(text_surface_arrival, (x + 10, y + 40))

        for i in range(self.env.bay_num):
            queue_x = i * (self.bay_width + 10) + 50
            fixed_title_y = self.screen_height - 450
            queue_text = f'Queue {i + 1}'
            text_surface = self.font.render(queue_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (queue_x, fixed_title_y))

            queue_y = fixed_title_y + 30 
            for index, vehicle in enumerate(self.env.waiting_queue[i]):
                y = queue_y + index * (self.queue_height + 5)
                queue_color = (100, 100, 255)
                pygame.draw.rect(self.screen, queue_color, (queue_x, y, self.queue_width, self.queue_height))

                vehicle_no = str(vehicle['ServiceNo'])
                arrival_time = vehicle['ActualArrival'].strftime('%H:%M:%S')
                vehicle_text = f"No: {vehicle_no}"
                arrival_text = f"Arr: {arrival_time}"
                text_surface_vehicle = self.font.render(vehicle_text, True, (255, 255, 255))
                text_surface_arrival = self.font.render(arrival_text, True, (255, 255, 255))

                self.screen.blit(text_surface_vehicle, (queue_x + 10, y + 5))
                self.screen.blit(text_surface_arrival, (queue_x + 10, y + 25))

        time_text = self.font.render(f'Time: {self.env.current_time}', True, (0, 0, 0))
        self.screen.blit(time_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


data = pd.read_excel('Data/preprocessed(simple).xlsx')
data = data[data['ActualArrival'].dt.date == pd.to_datetime('2024-06-19').date()]


model = PPO.load('Model/PPO_model.zip')

env = BusEnv(data)

visualizer = PygameVisualizer(env)

obs, _ = env.reset()

done = False
while not done:
    action, _ = model.predict(obs)
    #action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    visualizer.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

visualizer.close()