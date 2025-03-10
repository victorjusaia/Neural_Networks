import gym
import numpy as np
import pygame
from gym import spaces
from stable_baselines3 import PPO

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        
        # Configuración de Pygame
        self.screen_width = 288
        self.screen_height = 512
        self.gravity = 1
        self.jump_strength = -10
        
        self.bird_x = 50
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        
        self.pipe_width = 50
        self.pipe_gap = 100
        self.pipe_x = self.screen_width
        self.pipe_height = np.random.randint(100, 300)
        
        self.done = False
        
        # Espacio de acciones: 0 (no saltar), 1 (saltar)
        self.action_space = spaces.Discrete(2)
        
        # Espacio de observaciones: Posición del pájaro y de los tubos
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -10]),
            high=np.array([self.screen_width, self.screen_height, self.screen_height, 10]),
            dtype=np.float32
        )
        
        # Inicializar Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
    
    def reset(self):
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.pipe_x = self.screen_width
        self.pipe_height = np.random.randint(100, 300)
        self.done = False
        return np.array([self.bird_y, self.pipe_x, self.pipe_height, self.bird_velocity], dtype=np.float32)
    
    def step(self, action):
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        self.pipe_x -= 5
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.screen_width
            self.pipe_height = np.random.randint(100, 300)
        
        # Verificar colisiones
        if self.bird_y <= 0 or self.bird_y >= self.screen_height or (
            self.pipe_x <= self.bird_x <= self.pipe_x + self.pipe_width and
            (self.bird_y <= self.pipe_height or self.bird_y >= self.pipe_height + self.pipe_gap)):
            self.done = True
            reward = -10
        else:
            reward = 1
        
        obs = np.array([self.bird_y, self.pipe_x, self.pipe_height, self.bird_velocity], dtype=np.float32)
        return obs, reward, self.done, {}
    
    def render(self, mode="human"):
        self.screen.fill((135, 206, 250))  # Color de fondo (azul cielo)
        
        # Dibujar el pájaro
        pygame.draw.circle(self.screen, (255, 255, 0), (self.bird_x, int(self.bird_y)), 10)
        
        # Dibujar los tubos
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, 0, self.pipe_width, self.pipe_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, self.pipe_height + self.pipe_gap, self.pipe_width, self.screen_height - (self.pipe_height + self.pipe_gap)))
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        pygame.quit()

# Entrenamiento con PPO
env = FlappyBirdEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Guardar el modelo
model.save("flappybird_ppo")

# Bucle para visualizar el modelo en acción
obs = env.reset()
done = False  # <-- Inicializa la variable done
while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

env.close()