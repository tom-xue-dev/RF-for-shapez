from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from ppo_model.ShapezEnv import ShapezEnv
from stable_baselines3.common.callbacks import BaseCallback
from ppo_model import getmap
import torch as th
from pathlib import Path
from typing import Optional
import time


from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

def _select_device(prefer_cuda: bool = True) -> str:
    try:
        return "cuda" if prefer_cuda and th.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_torch_load_patched = False

def _patch_torch_load_weights_only():
    """Patch torch.serialization.load to ignore unknown 'weights_only' kwarg.
    Some older torch versions do not support this kwarg used by SB3.
    """
    global _torch_load_patched
    if _torch_load_patched:
        return
    try:
        import torch.serialization as _ts  # type: ignore
        _orig = _ts.load

        def _wrapped(*args, **kwargs):  # type: ignore
            if 'weights_only' in kwargs:
                kwargs.pop('weights_only')
            return _orig(*args, **kwargs)

        _ts.load = _wrapped  # type: ignore
        try:
            import torch as _troot  # type: ignore
            _troot.load = _wrapped  # ensure torch.load also patched
        except Exception:
            pass
        _torch_load_patched = True
    except Exception:
        # If patching fails, proceed; load may still work on newer torch
        pass


class ActionMaskCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ActionMaskCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        action_mask = self.env.get_action_mask()
        self.model.policy.set_action_mask(action_mask)
        return True
class MaskedMultiInputPolicy(MultiInputPolicy):
    def __init__(self, *args, model=None, **kwargs):
        super(MaskedMultiInputPolicy, self).__init__(*args, **kwargs)
        self.model = model  # 保存 model
        # default to no action mask until callback provides one
        self.action_mask = None


    def set_action_mask(self, action_mask):
        self.action_mask = action_mask

    def _get_original_env(self):
        env = None
        try:
            env = self.model.get_env()  # type: ignore[attr-defined]
        except Exception:
            env = None
        if env is None:
            env = getattr(self.model, 'env', None)
        e = env
        # unwrap VecNormalize / VecEnvWrapper -> DummyVecEnv -> base env
        depth_guard = 0
        while hasattr(e, 'venv') and depth_guard < 10:
            e = getattr(e, 'venv')
            depth_guard += 1
        if hasattr(e, 'envs') and e.envs:
            base = e.envs[0]
        else:
            base = e
        return getattr(base, 'unwrapped', base)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes, with optional action masking.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # IMPORTANT: Do NOT recompute mask from live env here, as it breaks
        # PPO training on replayed batches (mask would mismatch old observations).
        # Expect mask to be set via callback during rollout, or remain None.
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            action_logits = mean_actions
            if self.action_mask is not None:
                # one_indices = [index for index, value in enumerate(self.action_mask) if value == 1]
                # print(f"Indices of 1s in action_mask: {one_indices}")
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32, device=action_logits.device)
                # support per-batch (B, A) or flat (A,) masks
                if action_mask_tensor.dim() == 1:
                    action_logits = action_logits + (action_mask_tensor - 1) * 1e6
                else:
                    action_logits = action_logits + (action_mask_tensor - 1) * 1e6
            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            action_logits = mean_actions
            if self.action_mask is not None:
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32, device=action_logits.device)
                if action_mask_tensor.dim() == 1:
                    action_logits = action_logits + (action_mask_tensor - 1) * 1e6
                else:
                    action_logits = action_logits + (action_mask_tensor - 1) * 1e6

            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")




# print(getmap.load_shared_arrays()[1])

def get_agent_act_list(model_path: Optional[str] = None):

    resource = getmap.load_shared_arrays()[0]
    build = getmap.load_shared_arrays()[1]
    target_shape = getmap.load_needed_shape()
    # print(resource)
    # print(np.array2string(build, max_line_width=200))
    env = ShapezEnv(build, resource, target_shape=target_shape)
    env.reset()
    act_list = env.action_list
    # model = PPO(MaskedMultiInputPolicy, env=env, verbose=1, policy_kwargs={'model': None},gamma=0.98)
    default_model = Path(__file__).resolve().parent / "ppo_model.zip"
    chosen = Path(model_path) if model_path else default_model
    if not chosen.exists():
        # fallback to default if provided path invalid
        chosen = default_model
    # ensure compatibility with older torch when loading sb3 zips
    _patch_torch_load_weights_only()
    device = _select_device()
    model = PPO.load(
        str(chosen),
        env=env,
        gamma=0.98,
        custom_objects={"policy_class": MaskedMultiInputPolicy},
        device=device,
    )

    model.set_env(env)
    model.policy.model = model
    # model.learn(total_timesteps=10000)
    # keep model path stable
    model.save(str(chosen))

    obs, info = env.reset()
    agent_act = []
    for step in range(10000):

        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action.item())
        agent_act.append(act_list[action])

        if truncated == True:
            env.reset()
            agent_act = []
            print("Truncated")
        elif done == True:
            print("Goal reached!", "Reward:", reward)
            print(obs["grid"])
            break
    return agent_act
# print(get_agent_act_list())
# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")


def train_model(total_timesteps: int = 10000,
                save_dir: Optional[str] = None,
                model_name: Optional[str] = None,
                gamma: float = 0.98,
                device: Optional[str] = None) -> str:
    """
    训练 PPO 模型并保存到目标目录。

    参数:
    - total_timesteps: 训练步数。
    - save_dir: 保存目录（默认保存到 ppo_model 目录下的 trained 子目录）。
    - model_name: 保存文件名（可选，未提供则自动按时间戳生成）。
    - gamma: 折扣系数。

    返回:
    - 已保存模型文件的完整路径。
    """
    # 载入来自游戏共享内存的地图与目标形状
    resource, build = getmap.load_shared_arrays()
    target_shape = getmap.load_needed_shape()

    # 构建环境
    env = ShapezEnv(build, resource, target_shape=target_shape)

    # 创建保存目录
    base_dir = Path(__file__).resolve().parent
    out_dir = Path(save_dir) if save_dir else (base_dir / "trained")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成保存文件名
    if model_name:
        filename = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"ppo_shapez_model_{ts}.zip"
    out_path = out_dir / filename

    # 创建模型（带掩码策略）
    use_device = device or _select_device()
    model = PPO(
        MaskedMultiInputPolicy,
        env=env,
        verbose=1,
        gamma=gamma,
        learning_rate=1e-4,
        ent_coef=0.01,
        clip_range=0.1,
        n_steps=2048,
        batch_size=128,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'model': None},
        device=use_device,
    )
    model.policy.model = model  # 让策略可访问 model 和 env

    # 可选：使用回调在每个 step 同步动作掩码
    callback = ActionMaskCallback(env)

    # 开始训练
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 保存模型
    model.save(str(out_path))

    return str(out_path)
