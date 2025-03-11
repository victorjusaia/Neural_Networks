import gym
from stable_baselines3 import DQN
import torch

# Intentamos utilizar la grafica si es posible para el entrenamiento
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")


def train_pong():
    # Crear el entorno de Pong
    env = gym.make("Pong-v4", render_mode=None)
    
    # Crear el modelo DQN con parámetros optimizados
    model = DQN("CnnPolicy", env, verbose=1, learning_rate=1e-5, buffer_size=100000, exploration_final_eps=0.001,
                target_update_interval=1000, train_freq=8, batch_size=64, gamma=0.995, device=device)
    
    print("Entrenando el modelo...")  # Mensaje inicial
    
    # Entrenar el modelo
    model.learn(total_timesteps=5000000) 
    
    print("Entrenamiento finalizado.")
    
    # Guardar el modelo
    model.save("pong_dqn_optimized")
    print("Modelo guardado como 'pong_dqn_optimized'")
    env.close()

def visualize_pong():
    # Cargar el entorno en modo visual
    env = gym.make("Pong-v4", render_mode="human")  

    # Cargar el modelo entrenado
    model = DQN.load("pong_dqn_optimized")
     # Nos aseguramos de cargarlo en GPU si es posible
    model.to(device) 
    # Mensaje de carga
    print("Cargando el modelo 'pong_dqn_optimized'...")  
    
    # Ajustar el reset() para que funcione correctamente con la nueva API de Gym
    obs, info = env.reset() 
    done = False
    print("Visualizando la IA jugando...")
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
    
    print("Juego finalizado.")
    env.close()
    
if __name__ == "__main__":
    train_pong()  # Ejecutar esta línea para entrenar el modelo
    visualize_pong()  # Ejecutar esta línea para ver la IA jugar
