from stable_baselines3 import PPO
from env import LinuxPrivEscEnv

env = LinuxPrivEscEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

model.save("privesc_ppo_model")
print("O'qitish tugadi, model saqlandi: privesc_ppo_model.zip")
