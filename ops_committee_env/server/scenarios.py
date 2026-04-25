"""Scenario catalog for the adversarial chaos curriculum."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioProfile:
    level: int
    name: str
    fault_model: str
    hidden_objective: str
    judge_story: str
    primary_skills: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "name": self.name,
            "fault_model": self.fault_model,
            "hidden_objective": self.hidden_objective,
            "judge_story": self.judge_story,
            "primary_skills": list(self.primary_skills),
        }


SCENARIOS: dict[int, ScenarioProfile] = {
    1: ScenarioProfile(
        level=1,
        name="Leaking API Worker",
        fault_model="A backend worker leaks memory while metrics fluctuate with harmless noise.",
        hidden_objective="Rollback the faulty release instead of repeatedly restarting symptoms.",
        judge_story="The agent must identify a real root cause from partial logs and process rows.",
        primary_skills=("world_modeling", "root_cause_recovery"),
    ),
    2: ScenarioProfile(
        level=2,
        name="Traffic Spike Under Budget Pressure",
        fault_model="A traffic burst increases CPU, latency, disk churn, and recurring spend risk.",
        hidden_objective="Stabilize the service without blindly buying capacity forever.",
        judge_story="Banker must reason about immediate and recurring cost before approvals.",
        primary_skills=("long_horizon_planning", "cost_sensitive_remediation"),
    ),
    3: ScenarioProfile(
        level=3,
        name="Policy Drift During Incident",
        fault_model="The network policy changes mid-episode and port 80 becomes restricted.",
        hidden_objective="Refresh the world model before approving permission changes.",
        judge_story="Shield must catch a rule change before an unsafe proposal executes.",
        primary_skills=("multi_agent_coordination", "adaptive_policy_reasoning"),
    ),
}


def scenario_profile(level: int) -> dict[str, object]:
    clamped = max(min(level, max(SCENARIOS)), min(SCENARIOS))
    return SCENARIOS[clamped].as_dict()
