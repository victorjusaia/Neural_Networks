import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def train_pong():
    # Crear el entorno de Pong
    env = gym.make("PongNoFrameskip-v4", render_mode=None)  
    # Crear el modelo DQN
    model = DQN("CnnPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=50000, exploration_final_eps=0.01, target_update_interval=1000, train_freq=4, batch_size=32, gamma=0.99)
    
    # Entrenar el modelo
    model.learn(total_timesteps=500000)
    
    # Guardar el modelo
    model.save("pong_dqn")
    env.close()

def visualize_pong():
    # Cargar el entorno en modo visual
    env = gym.make("ALE/Pong-v5", render_mode="human")
    
    # Cargar el modelo entrenado
    model = DQN.load("pong_dqn")
    
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
    
    env.close()

if __name__ == "__main__":
    train_pong()  # Ejecutar esta línea para entrenar el modelo
    visualize_pong()  # Ejecutar esta línea para ver la IA jugar
