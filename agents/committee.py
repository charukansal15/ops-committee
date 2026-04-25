"""Rule-based committee policies used for demos and baseline comparisons."""

from __future__ import annotations

import random
from typing import Any

from ops_committee_env.models import (
    AgentRole,
    AuditDecision,
    OpsCommitteeAction,
    OpsCommitteeObservation,
    OpsTool,
)
from ops_committee_env.server.ops_committee_environment import OpsCommitteeEnvironment


def _dump_observation(obs: OpsCommitteeObservation) -> dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return {
        key: value
        for key, value in obs.__dict__.items()
        if not key.startswith("_")
    }


def _proposal_id(obs: OpsCommitteeObservation) -> str | None:
    if not obs.pending_proposal:
        return None
    return str(obs.pending_proposal["proposal_id"])


def choose_banker_audit(obs: OpsCommitteeObservation) -> AuditDecision:
    impact = (obs.pending_proposal or {}).get("impact_estimates", {})
    if impact.get("terminal_budget_risk"):
        return AuditDecision.VETO
    if impact.get("budget_post_action") == "critical_risk":
        return AuditDecision.VETO
    return AuditDecision.APPROVE


def choose_shield_audit(obs: OpsCommitteeObservation) -> AuditDecision:
    impact = (obs.pending_proposal or {}).get("impact_estimates", {})
    if impact.get("policy_conflict_detected"):
        return AuditDecision.VETO
    return AuditDecision.APPROVE


def choose_fixer_action(obs: OpsCommitteeObservation) -> OpsCommitteeAction:
    metrics = obs.visible_metrics
    policy = obs.policy
    process_rows = obs.process_table

    if metrics.get("security") in {"risky", "critical"} and policy.get("restricted_ports"):
        port = str(policy["restricted_ports"][0])
        return OpsCommitteeAction(
            actor=AgentRole.FIXER,
            tool=OpsTool.MODIFY_PERMISSIONS,
            target=port,
            params={"mode": "close_port", "port": int(port)},
            rationale="Close the exposed restricted port before further remediation.",
        )

    if metrics.get("disk") in {"elevated", "critical"}:
        return OpsCommitteeAction(
            actor=AgentRole.FIXER,
            tool=OpsTool.CLEANUP_DISK,
            rationale="Disk pressure is visible; clean safe temporary data first.",
        )

    leaking_api = any(
        row.get("service") == "api" and row.get("memory_trend") == "rising"
        for row in process_rows
    )
    if leaking_api or metrics.get("memory") in {"elevated", "critical"}:
        return OpsCommitteeAction(
            actor=AgentRole.FIXER,
            tool=OpsTool.ROLLBACK_RELEASE,
            rationale="API memory is rising, so rollback targets the likely root cause.",
        )

    if metrics.get("cpu") in {"elevated", "critical"} or metrics.get("latency") in {
        "elevated",
        "critical",
    }:
        amount = 2 if "critical" in {metrics.get("cpu"), metrics.get("latency")} else 1
        return OpsCommitteeAction(
            actor=AgentRole.FIXER,
            tool=OpsTool.SCALE_INFRASTRUCTURE,
            amount=amount,
            rationale="Add bounded capacity after visible CPU or latency pressure appears.",
        )

    return OpsCommitteeAction(
        actor=AgentRole.FIXER,
        tool=OpsTool.CHECK_LOGS,
        rationale="Gather fresh logs before proposing more changes.",
    )


def _next_committee_action(obs: OpsCommitteeObservation) -> OpsCommitteeAction:
    status = obs.committee_status
    missing = status.get("missing_approvals", [])
    proposal_id = _proposal_id(obs)
    if obs.pending_proposal and "banker" in missing:
        decision = choose_banker_audit(obs)
        return OpsCommitteeAction(
            actor=AgentRole.BANKER,
            tool=OpsTool.AUDIT_PROPOSAL,
            proposal_id=proposal_id,
            decision=decision,
            rationale=(
                "Impact report would exhaust or critically constrain the budget."
                if decision is AuditDecision.VETO
                else "Impact report fits the current cost envelope."
            ),
        )
    if obs.pending_proposal and "shield" in missing:
        decision = choose_shield_audit(obs)
        return OpsCommitteeAction(
            actor=AgentRole.SHIELD,
            tool=OpsTool.AUDIT_PROPOSAL,
            proposal_id=proposal_id,
            decision=decision,
            rationale=(
                "Impact report shows a policy conflict."
                if decision is AuditDecision.VETO
                else "No permission conflict is visible in the impact report."
            ),
        )
    return choose_fixer_action(obs)


def run_rule_based_committee(
    *,
    level: int = 3,
    seed: int = 123,
    max_steps: int = 20,
) -> dict[str, Any]:
    env = OpsCommitteeEnvironment()
    obs = env.reset(level=level, seed=seed, max_steps=max_steps)
    trajectory: list[dict[str, Any]] = [{"observation": _dump_observation(obs)}]

    while not obs.done and env.state.step_count < max_steps:
        action = _next_committee_action(obs)
        obs = env.step(action)
        trajectory.append(
            {
                "action": action.model_dump() if hasattr(action, "model_dump") else action.__dict__,
                "observation": _dump_observation(obs),
            }
        )

    return {
        "policy": "rule_based_committee",
        "level": level,
        "seed": seed,
        "terminal_reason": env.state.terminal_reason,
        "steps": env.state.step_count,
        "total_reward": env._world.last_reward_breakdown.get("total", 0.0),
        "trajectory": trajectory,
    }


def run_random_committee(
    *,
    level: int = 3,
    seed: int = 123,
    max_steps: int = 20,
) -> dict[str, Any]:
    rng = random.Random(seed)
    env = OpsCommitteeEnvironment()
    obs = env.reset(level=level, seed=seed, max_steps=max_steps)
    mutating_tools = [
        OpsTool.RESTART_SERVICE,
        OpsTool.SCALE_INFRASTRUCTURE,
        OpsTool.MODIFY_PERMISSIONS,
        OpsTool.ROLLBACK_RELEASE,
        OpsTool.CLEANUP_DISK,
        OpsTool.APPLY_RATE_LIMIT,
    ]
    trajectory: list[dict[str, Any]] = [{"observation": _dump_observation(obs)}]

    while not obs.done and env.state.step_count < max_steps:
        if obs.pending_proposal:
            missing = obs.committee_status.get("missing_approvals", [])
            actor = AgentRole.BANKER if "banker" in missing else AgentRole.SHIELD
            action = OpsCommitteeAction(
                actor=actor,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=_proposal_id(obs),
                decision=AuditDecision.APPROVE,
                rationale="Baseline approves pending actions without impact analysis.",
            )
        else:
            tool = rng.choice(mutating_tools)
            params: dict[str, Any] = {}
            target = None
            amount = rng.randint(1, 5)
            if tool == OpsTool.RESTART_SERVICE:
                target = rng.choice(["frontend", "api", "db", "cache"])
            elif tool == OpsTool.MODIFY_PERMISSIONS:
                port = rng.choice([22, 80, 443, 8080])
                target = str(port)
                params = {"mode": rng.choice(["open_port", "enable_root"]), "port": port}
            action = OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=tool,
                target=target,
                amount=amount,
                params=params,
                rationale="Random baseline action.",
            )
        obs = env.step(action)
        trajectory.append(
            {
                "action": action.model_dump() if hasattr(action, "model_dump") else action.__dict__,
                "observation": _dump_observation(obs),
            }
        )

    return {
        "policy": "random_approve_all",
        "level": level,
        "seed": seed,
        "terminal_reason": env.state.terminal_reason,
        "steps": env.state.step_count,
        "total_reward": env._world.last_reward_breakdown.get("total", 0.0),
        "trajectory": trajectory,
    }
