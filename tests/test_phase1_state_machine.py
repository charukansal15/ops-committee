import unittest

from ops_committee_env.models import AgentRole, AuditDecision, OpsCommitteeAction, OpsTool
from ops_committee_env.server.ops_committee_environment import OpsCommitteeEnvironment
from ops_committee_env.server.system_state import SimConfig, SystemState


def approve_pending_proposal(env: OpsCommitteeEnvironment):
    proposal_id = env.state.pending_proposal_id
    env.step(
        OpsCommitteeAction(
            actor=AgentRole.BANKER,
            tool=OpsTool.AUDIT_PROPOSAL,
            proposal_id=proposal_id,
            decision=AuditDecision.APPROVE,
            rationale="Cost impact is acceptable.",
        )
    )
    return env.step(
        OpsCommitteeAction(
            actor=AgentRole.SHIELD,
            tool=OpsTool.AUDIT_PROPOSAL,
            proposal_id=proposal_id,
            decision=AuditDecision.APPROVE,
            rationale="Security risk is acceptable.",
        )
    )


class Phase1StateMachineTests(unittest.TestCase):
    def test_reset_returns_partial_observability(self) -> None:
        env = OpsCommitteeEnvironment()
        obs = env.reset(level=1, episode_id="test-episode")

        self.assertEqual(env.state.episode_id, "test-episode")
        self.assertIn("cpu", obs.visible_metrics)
        self.assertIn(obs.visible_metrics["cpu"], {"nominal", "elevated", "critical"})
        self.assertNotIn("cpu_usage", obs.visible_metrics)
        self.assertGreaterEqual(len(obs.process_table), 3)

    def test_memory_leak_progresses_without_root_fix(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=1)
        before = env._world.ram_usage

        obs = env.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))

        self.assertGreater(env._world.ram_usage, before)
        self.assertFalse(obs.done)
        self.assertIn("memory", obs.visible_metrics)

    def test_rollback_fixes_memory_leak_and_rewards_progress(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=1)

        env.step(OpsCommitteeAction(tool=OpsTool.ROLLBACK_RELEASE))
        obs = approve_pending_proposal(env)

        self.assertFalse(env._world.memory_leak_active)
        self.assertGreater(obs.reward_breakdown["progress"], 0)

    def test_level_three_policy_drift_changes_visible_policy(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=3, max_steps=10)

        obs = None
        for _ in range(5):
            obs = env.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))

        self.assertIsNotNone(obs)
        self.assertEqual(env.state.policy_version, 2)
        self.assertIn(80, obs.policy["restricted_ports"])

    def test_insecure_permission_change_gets_heavy_penalty(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=3)

        env.step(
            OpsCommitteeAction(
                tool=OpsTool.MODIFY_PERMISSIONS,
                target="22",
                params={"mode": "open_port", "port": 22},
            )
        )
        obs = approve_pending_proposal(env)

        self.assertLess(obs.reward_breakdown["security"], 0)
        self.assertEqual(env.state.security_posture, "risky")

    def test_seeded_jitter_is_reproducible(self) -> None:
        first = OpsCommitteeEnvironment()
        second = OpsCommitteeEnvironment()
        first.reset(level=2, seed=123)
        second.reset(level=2, seed=123)

        first.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))
        second.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))

        self.assertEqual(round(first._world.cpu_usage, 4), round(second._world.cpu_usage, 4))
        self.assertEqual(round(first._world.ram_usage, 4), round(second._world.ram_usage, 4))

    def test_different_seeds_change_metric_trajectory(self) -> None:
        first = OpsCommitteeEnvironment()
        second = OpsCommitteeEnvironment()
        first.reset(level=2, seed=123)
        second.reset(level=2, seed=456)

        first.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))
        second.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))

        self.assertNotEqual(round(first._world.cpu_usage, 4), round(second._world.cpu_usage, 4))

    def test_process_table_includes_harmless_noise(self) -> None:
        env = OpsCommitteeEnvironment()
        obs = env.reset(level=1, seed=123)

        services = {row["service"] for row in obs.process_table}
        self.assertTrue({"frontend", "api", "db"}.issubset(services))
        self.assertTrue(any(row.get("noise") for row in obs.process_table))

    def test_veto_feedback_is_exposed_in_observation(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=3, seed=123)

        env.step(
            OpsCommitteeAction(
                tool=OpsTool.MODIFY_PERMISSIONS,
                params={"mode": "enable_root"},
            )
        )
        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=env.state.pending_proposal_id,
                decision=AuditDecision.VETO,
                rationale="Action violates least-privilege policy.",
            )
        )

        self.assertTrue(obs.pending_feedback)
        self.assertIn("Shield Veto", obs.pending_feedback[0])

    def test_invalid_permission_target_does_not_crash_episode(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=1, seed=123)

        env.step(
            OpsCommitteeAction(
                tool=OpsTool.MODIFY_PERMISSIONS,
                target="api",
                params={"mode": "open_port"},
            )
        )
        obs = approve_pending_proposal(env)

        self.assertFalse(obs.done)
        self.assertIn("Invalid port target", obs.message)
        self.assertTrue(obs.pending_feedback)
        self.assertLess(obs.reward_breakdown["anti_gaming"], 0)

    def test_sim_config_controls_scale_cost(self) -> None:
        config = SimConfig(scale_unit_cost=99.0, recurring_cost_per_resource=0.0)
        state = SystemState.initial("config-test", seed=123, config=config)

        state.apply("scale_infrastructure", None, 1, {})

        self.assertEqual(state.budget_credits, 901.0)

    def test_feedback_accumulates_until_observation_consumes_it(self) -> None:
        state = SystemState.initial("feedback-test", seed=123)
        state.pending_feedback.append("System Warning: policy changed outside action loop.")

        state.apply("modify_permissions", "api", 1, {"mode": "open_port"})
        first_observation = state.observe()
        second_observation = state.observe()

        self.assertEqual(len(first_observation["pending_feedback"]), 2)
        self.assertEqual(second_observation["pending_feedback"], [])

    def test_policy_drift_adds_system_level_feedback(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(level=3, seed=123, max_steps=10)

        obs = None
        for _ in range(5):
            obs = env.step(OpsCommitteeAction(tool=OpsTool.CHECK_LOGS))

        self.assertIsNotNone(obs)
        self.assertTrue(obs.pending_feedback)
        self.assertIn("policy drift", obs.pending_feedback[0])

    def test_metric_thresholds_are_config_driven(self) -> None:
        config = SimConfig(
            cpu_warn_threshold=10.0,
            cpu_critical_threshold=20.0,
            cpu_pressure_weight=1.0,
            memory_pressure_weight=0.0,
            disk_pressure_weight=0.0,
            latency_pressure_weight=0.0,
            error_pressure_weight=0.0,
        )
        state = SystemState.initial("threshold-test", seed=123, config=config)
        state.cpu_usage = 25.0
        state.ram_usage = 0.0
        state.disk_usage = 0.0
        state.latency_ms = 0.0
        state.error_rate_pct = 0.0

        state._recompute_health()
        observation = state.observe()

        self.assertEqual(observation["visible_metrics"]["cpu"], "critical")
        self.assertEqual(state.service_health, 85.0)


if __name__ == "__main__":
    unittest.main()
