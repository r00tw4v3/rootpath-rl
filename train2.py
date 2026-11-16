from stable_baselines3 import PPO
from env_v2 import LinuxPrivEscEnvV2, VULN_NAMES

model = PPO.load("privesc_ppo_v2")
env = LinuxPrivEscEnvV2()

obs, _ = env.reset()
print("Haqiqiy vulnlar:", [VULN_NAMES[i] for i in env.real_vulns])
for step in range(15):
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)          # <-- shu qator qo'shildi
    name = VULN_NAMES[action % 8]
    kind = "CHECK" if action < 8 else "EXPLOIT"
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Qadam {step}: {kind} {name}, reward={reward}")
    if terminated or truncated:
        break
