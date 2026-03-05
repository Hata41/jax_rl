from jax_rl.configs.config import ExperimentConfig, register_configs
from jax_rl.systems.alphazero.eval import EvaluationManager


def test_alphazero_eval_manager_action_selection_default_and_override():
    captured = {}

    class _FakeEvaluator:
        def __init__(
            self,
            config,
            env_name,
            num_episodes,
            max_steps_per_episode,
            greedy,
            env_kwargs=None,
            action_selection="policy",
        ):
            del config, env_name, num_episodes, max_steps_per_episode, greedy, env_kwargs
            captured.setdefault("modes", []).append(action_selection)

        def run(self, replicated_params, seed):
            del replicated_params, seed
            return {
                "return_mean": 0.0,
                "return_std": 0.0,
                "return_min": 0.0,
                "return_max": 0.0,
                "episodes": 1,
                "steps": 1,
            }

        def close(self):
            return

    cfg = ExperimentConfig()
    manager = EvaluationManager(
        config=cfg,
        evaluations={
            "default_mode": {
                "env_name": "jaxpallet:PMC-PLD",
                "num_episodes": 1,
                "eval_every": 1,
            },
            "search_mode": {
                "env_name": "jaxpallet:PMC-PLD",
                "num_episodes": 1,
                "eval_every": 1,
                "action_selection": "search",
            },
        },
        default_env_name=cfg.env.env_name,
        default_env_kwargs=cfg.env.env_kwargs,
        evaluator_cls=_FakeEvaluator,
    )
    try:
        assert captured["modes"] == ["policy", "search"]
    finally:
        manager.close()


def test_hydra_compose_loads_nested_alphazero_config_with_eval_modes():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    register_configs()
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="alphazero/train_jaxpallet")

    obj = OmegaConf.to_object(cfg)
    if isinstance(obj, dict) and "alphazero" in obj:
        obj = obj["alphazero"]
    assert isinstance(obj, ExperimentConfig)
    assert obj.system.name == "alphazero"
    assert obj.evaluations["policy_eval"]["action_selection"] == "policy"
    assert obj.evaluations["search_eval"]["action_selection"] == "search"
