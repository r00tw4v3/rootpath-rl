from stable_baselines3 import PPO
from env_v2 import LinuxPrivEscEnvV2

env = LinuxPrivEscEnvV2()
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=100000)
model.save("privesc_ppo_v2")
print("O'qitish tugadi: privesc_ppo_v2.zip")

