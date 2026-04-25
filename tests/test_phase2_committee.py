import unittest

from ops_committee_env.models import (
    AgentRole,
    AuditDecision,
    OpsCommitteeAction,
    OpsTool,
)
from ops_committee_env.server.ops_committee_environment import OpsCommitteeEnvironment
from ops_committee_env.server.tools import RESERVED_TOOL_NAMES


class Phase2CommitteeTests(unittest.TestCase):
    def test_tool_registry_excludes_reserved_names(self) -> None:
        env = OpsCommitteeEnvironment()
        tool_names = {tool["name"] for tool in env.list_tools()}

        self.assertFalse(tool_names.intersection(RESERVED_TOOL_NAMES))
        self.assertIn(OpsTool.CHECK_LOGS.value, tool_names)
        self.assertIn(OpsTool.AUDIT_PROPOSAL.value, tool_names)

    def test_tool_registry_exposes_billing_metadata(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)

        tools = {tool["name"]: tool for tool in env.list_tools()}

        scale_cost_model = tools[OpsTool.SCALE_INFRASTRUCTURE.value]["cost_model"]
        self.assertEqual(scale_cost_model["cost_per_unit"], env._world.config.scale_unit_cost)
        self.assertEqual(
            scale_cost_model["recurring_cost_per_resource"],
            env._world.config.recurring_cost_per_resource,
        )

    def test_fixer_mutating_action_becomes_pending_proposal(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        initial_units = env._world.resource_units

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=2,
                rationale="Reduce pressure before user-visible errors rise.",
            )
        )

        self.assertEqual(env._world.resource_units, initial_units)
        self.assertIsNotNone(obs.pending_proposal)
        self.assertEqual(obs.committee_status["phase"], "awaiting_audits")
        self.assertEqual(set(obs.committee_status["missing_approvals"]), {"banker", "shield"})

    def test_scale_proposal_exposes_budget_impact_before_audit(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env._world.budget_credits = 100.0

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=5,
            )
        )

        impact = obs.pending_proposal["impact_estimates"]
        self.assertEqual(impact["estimated_immediate_cost"], 140.0)
        self.assertTrue(impact["terminal_budget_risk"])
        self.assertEqual(impact["budget_post_action"], "terminal_risk")
        self.assertEqual(obs.billing_manifest["scale_unit_cost"], 28.0)
        self.assertIn("banker", obs.committee_handbook["audit_responsibilities"])

    def test_permission_proposal_exposes_policy_conflict_before_audit(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env._world.restricted_ports.add(80)

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.MODIFY_PERMISSIONS,
                target="80",
                params={"mode": "open_port", "port": 80},
            )
        )

        impact = obs.pending_proposal["impact_estimates"]
        self.assertTrue(impact["policy_conflict_detected"])
        self.assertEqual(impact["target_port"], 80)
        self.assertIn(80, impact["restricted_ports"])

    def test_final_approval_executes_pending_proposal(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=2,
            )
        )
        proposal_id = env.state.pending_proposal_id

        banker_obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="Two units fit the current incident budget.",
            )
        )
        self.assertEqual(env._world.resource_units, 1)
        self.assertEqual(banker_obs.committee_status["missing_approvals"], ["shield"])

        shield_obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="No permission boundary changes are involved.",
            )
        )

        self.assertEqual(env._world.resource_units, 3)
        self.assertEqual(shield_obs.committee_status["phase"], "ready")
        self.assertTrue(any("Committee Approved" in item for item in shield_obs.pending_feedback))

    def test_veto_blocks_execution_and_clears_pending_proposal(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.MODIFY_PERMISSIONS,
                params={"mode": "enable_root"},
                rationale="Try to unblock permissions quickly.",
            )
        )
        proposal_id = env.state.pending_proposal_id

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.VETO,
                rationale="Root access violates least privilege.",
            )
        )

        self.assertIsNone(env.state.pending_proposal_id)
        self.assertEqual(env.state.security_posture, "clean")
        self.assertIn("Shield Veto", obs.pending_feedback[0])

    def test_non_fixer_cannot_propose_mutating_operation(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=1,
            )
        )

        self.assertIsNone(env.state.pending_proposal_id)
        self.assertIn("not allowed", obs.message)
        self.assertLess(obs.reward_breakdown["anti_gaming"], 0)

    def test_audit_without_pending_proposal_is_penalized(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.AUDIT_PROPOSAL,
                decision=AuditDecision.APPROVE,
            )
        )

        self.assertIn("No proposal", obs.message)
        self.assertLess(obs.reward_breakdown["anti_gaming"], 0)

    def test_negligent_banker_penalty_on_budget_exhaustion(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env._world.budget_credits = 100.0
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=5,
            )
        )
        proposal_id = env.state.pending_proposal_id
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="Approving despite budget risk.",
            )
        )
        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="No policy change.",
            )
        )

        self.assertEqual(env.state.terminal_reason, "budget_exhausted")
        self.assertLess(obs.reward_breakdown["negligent_audit_banker"], 0)
        self.assertEqual(obs.reward_breakdown["negligent_audit_shield"], 0.0)

    def test_negligent_shield_penalty_on_security_breach(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.MODIFY_PERMISSIONS,
                params={"mode": "enable_root"},
            )
        )
        proposal_id = env.state.pending_proposal_id
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="No direct spend.",
            )
        )
        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="Approving despite root access.",
            )
        )

        self.assertEqual(env.state.terminal_reason, "security_breach")
        self.assertLess(obs.reward_breakdown["negligent_audit_shield"], 0)
        self.assertEqual(obs.reward_breakdown["negligent_audit_banker"], 0.0)

    def test_non_mutating_tool_rationale_is_broadcast(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.DO_NOTHING,
                rationale="I need to inspect budget impact before approving.",
            )
        )

        self.assertTrue(
            any("inspect budget impact" in item for item in obs.pending_feedback)
        )
        self.assertTrue(
            any(
                event.source == "committee_chat"
                and "inspect budget impact" in event.message
                for event in env._world.events
            )
        )

    def test_committee_execution_event_preserves_proposal_rationale(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=1,
                rationale="Scale one unit to avoid overpaying for capacity.",
            )
        )
        proposal_id = env.state.pending_proposal_id
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.BANKER,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="Budget can absorb one unit.",
            )
        )
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="No permission boundary is touched.",
            )
        )

        execution_events = [
            event
            for event in env._world.events
            if event.source == "committee"
            and "executing proposal" in event.message
        ]
        self.assertEqual(len(execution_events), 1)
        self.assertIn("avoid overpaying", execution_events[0].data["rationale"])

    def test_duplicate_proposal_feedback_names_missing_auditor(self) -> None:
        env = OpsCommitteeEnvironment()
        env.reset(seed=123)
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=1,
            )
        )
        env.step(
            OpsCommitteeAction(
                actor=AgentRole.SHIELD,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=env.state.pending_proposal_id,
                decision=AuditDecision.APPROVE,
                rationale="No security issue.",
            )
        )

        obs = env.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=1,
            )
        )

        self.assertTrue(obs.pending_feedback)
        self.assertIn("Proposal prop-0001", obs.pending_feedback[0])
        self.assertIn("banker", obs.pending_feedback[0])


if __name__ == "__main__":
    unittest.main()
