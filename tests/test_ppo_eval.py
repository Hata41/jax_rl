from typing import Any, cast

from jax_rl.systems.ppo.eval import EvaluationManager


def test_ppo_eval_manager_is_lazy_and_respects_schedule():
    captured = {"constructed": [], "runs": []}

    class _FakeEvaluator:
        def __init__(
            self,
            env_name,
            num_episodes,
            max_steps_per_episode,
            greedy,
            env_kwargs=None,
        ):
            del max_steps_per_episode, greedy
            captured["constructed"].append((env_name, num_episodes, dict(env_kwargs or {})))

        def run(self, replicated_params, seed):
            del replicated_params
            captured["runs"].append(seed)
            return {
                "return_mean": 0.1,
                "return_std": 0.0,
                "return_min": 0.1,
                "return_max": 0.1,
                "episodes": 2,
                "steps": 4,
            }

        def close(self):
            return

    manager = EvaluationManager(
        evaluations={
            "default_eval": {
                "env_name": "rlpallet:UldEnv-v2",
                "num_episodes": 2,
                "eval_every": 2,
                "env_kwargs": {"max_items": 10},
            }
        },
        default_env_name="rlpallet:UldEnv-v2",
        default_env_kwargs={"max_items": 10},
        evaluator_cls=cast(Any, _FakeEvaluator),
        now_fn=lambda: 0.0,
    )

    try:
        assert captured["constructed"] == []

        skipped = manager.run_if_needed(update_idx=1, params=None, seed=11)
        assert skipped == {}
        assert captured["constructed"] == []
        assert captured["runs"] == []

        ran = manager.run_if_needed(update_idx=2, params=None, seed=22)
        assert "default_eval/return_mean" in ran
        assert captured["constructed"] == [
            ("rlpallet:UldEnv-v2", 2, {"max_items": 10})
        ]
        assert captured["runs"] == [22]

        manager.run_if_needed(update_idx=4, params=None, seed=44)
        assert len(captured["constructed"]) == 1
        assert captured["runs"] == [22, 44]
    finally:
        manager.close()
