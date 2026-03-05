from jax_rl.configs.config import EnvConfig, ExperimentConfig, SystemConfig
from jax_rl.envs.env import make_stoa_env
from jax_rl.systems.ppo.anakin.factory import build_system
from jax_rl.systems.ppo.anakin.steps import make_ppo_steps
from jax_rl.systems.ppo.update import make_actor_optimizer, make_critic_optimizer


def test_make_ppo_steps_returns_pmap_functions():
    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1"),
        system=SystemConfig(num_envs=1, num_steps=4, minibatch_size=4),
    )
    env, env_params = make_stoa_env(config.env.env_name, num_envs_per_device=1)
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)

    pmap_rollout, pmap_update = make_ppo_steps(
        config=config,
        env=env,
        env_params=env_params,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    assert callable(pmap_rollout)
    assert callable(pmap_update)
    assert hasattr(pmap_rollout, "lower")
    assert hasattr(pmap_update, "lower")


def test_build_system_is_exposed_from_factory_module():
    assert callable(build_system)
