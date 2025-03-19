import time
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from network.custom_network import CustomACNetwork

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 8     # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=CustomACNetwork,
        features_extractor_kwargs=dict(features_dim=64),
    )

    model = PPO(policy="MlpPolicy",
                policy_kwargs=policy_kwargs,
                env=vec_env,
                learning_rate=2.5e-4,
                batch_size=32,
                device="cpu",
                tensorboard_log="./tb_logs/",
                verbose=1)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    callbacks = []
    eval_callback = EvalCallback(
        vec_env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path="./",
        log_path="./tb_logs/training_" + time_stamp + "_eval",
        eval_freq=1000,
        verbose=1
    )
    callbacks.append(eval_callback)

    kwargs = {}
    kwargs["callback"] = callbacks

    model.learn(total_timesteps=50_000,
                tb_log_name="policy_" + time_stamp,
                **kwargs)

    model.save("mars_" + time_stamp)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()