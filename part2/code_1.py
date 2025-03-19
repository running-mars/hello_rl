import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from network.custom_network import CustomACNetwork

# env set
env = gym.make("CartPole-v1", render_mode="rgb_array")

# network set
policy_kwargs = dict(
        features_extractor_class=CustomACNetwork,
        features_extractor_kwargs=dict(features_dim=64),
    )

# rl model with algorithms and network
model = PPO(policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            env=env,
            learning_rate=2.5e-4,
            batch_size=32,
            device="cpu",
            tensorboard_log="./tb_logs/",
            verbose=1)

time_stamp = time.strftime("%Y%m%d-%H%M%S")

callbacks = []
eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path="../part1/",
        log_path="./tb_logs/mars_" + time_stamp + "_eval",
        eval_freq=1000,
        verbose=1
    )
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# training and then save model parameters
model.learn(total_timesteps=50_000,
            tb_log_name="mars_" + time_stamp,
            **kwargs)

model.save("mars_" + time_stamp)

# evaluation after training
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
