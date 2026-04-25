"""Backend-system simulator for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from ops_committee_env.server.scenarios import scenario_profile


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _band(value: float, warn: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warn:
        return "elevated"
    return "nominal"


@dataclass
class Event:
    step: int
    level: str
    source: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class ActionOutcome:
    executed: bool
    message: str
    cost: float = 0.0
    safety_violation: bool = False
    root_cause_fixed: bool = False
    anti_gaming_penalty: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    pending_feedback: list[str] = field(default_factory=list)


@dataclass
class CommitteeAudit:
    actor: str
    decision: str
    reason: str
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "decision": self.decision,
            "reason": self.reason,
            "step": self.step,
        }


@dataclass
class PendingProposal:
    proposal_id: str
    proposer: str
    tool: str
    target: str | None
    amount: int
    params: dict[str, Any]
    rationale: str
    created_step: int
    impact_estimates: dict[str, Any] = field(default_factory=dict)
    audits: dict[str, CommitteeAudit] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "tool": self.tool,
            "target": self.target,
            "amount": self.amount,
            "params": self.params,
            "rationale": self.rationale,
            "created_step": self.created_step,
            "impact_estimates": self.impact_estimates,
            "audits": {role: audit.as_dict() for role, audit in self.audits.items()},
        }


@dataclass(frozen=True)
class SimConfig:
    """Tunable simulation and reward constants for fast reward shaping."""

    min_chaos_level: int = 1
    max_chaos_level: int = 3
    max_action_amount: int = 10
    feedback_queue_limit: int = 12
    committee_required_approvers: frozenset[str] = frozenset({"banker", "shield"})
    proposal_timeout_steps: int = 4
    allowed_public_ports: frozenset[int] = frozenset({443, 8080})
    invalid_action_penalty: float = 8.0
    invalid_target_penalty: float = 10.0
    invalid_committee_penalty: float = 8.0
    negligent_audit_penalty: float = 35.0
    veto_penalty: float = 4.0
    stale_proposal_penalty: float = 5.0
    no_action_penalty: float = 2.0
    unknown_service_penalty: float = 6.0
    restart_cost: float = 4.0
    scale_unit_cost: float = 28.0
    scale_spam_threshold: int = 4
    scale_spam_penalty: float = 18.0
    close_port_cost: float = 1.0
    root_access_penalty: float = 25.0
    unsafe_port_penalty: float = 20.0
    rollback_cost: float = 16.0
    cleanup_disk_cost: float = 3.0
    rotate_secret_cost: float = 5.0
    rate_limit_cost: float = 2.0
    recurring_cost_per_resource: float = 2.0
    uptime_reward_divisor: float = 10.0
    action_cost_weight: float = 0.12
    extra_resource_penalty: float = 0.6
    safety_violation_penalty: float = 100.0
    open_security_flag_penalty: float = 18.0
    root_cause_progress_reward: float = 18.0
    clean_recovery_bonus: float = 45.0
    terminal_failure_penalty: float = 60.0
    step_penalty: float = 1.0
    clean_recovery_min_budget: float = 700.0
    budget_critical_threshold: float = 120.0
    budget_constrained_threshold: float = 350.0
    cpu_warn_threshold: float = 65.0
    cpu_critical_threshold: float = 88.0
    memory_warn_threshold: float = 70.0
    memory_degraded_threshold: float = 72.0
    memory_critical_threshold: float = 90.0
    disk_warn_threshold: float = 75.0
    disk_pressure_threshold: float = 80.0
    disk_critical_threshold: float = 92.0
    latency_warn_threshold: float = 260.0
    latency_critical_threshold: float = 700.0
    error_warn_threshold: float = 2.5
    error_critical_threshold: float = 8.0
    health_warn_loss_threshold: float = 20.0
    health_critical_loss_threshold: float = 55.0
    cpu_pressure_weight: float = 0.45
    memory_pressure_weight: float = 0.55
    disk_pressure_weight: float = 0.35
    latency_pressure_weight: float = 0.04
    error_pressure_weight: float = 2.0
    overload_error_increase: float = 3.0
    overload_latency_increase: float = 90.0
    leak_error_increase: float = 0.4
    recovery_error_decrease: float = 1.5
    recovery_latency_floor: float = 90.0
    recovery_latency_decrease: float = 55.0
    catastrophic_health_threshold: float = 10.0
    clean_recovery_min_health: float = 90.0
    background_noise_probability: float = 0.55
    alert_cpu_threshold: float = 80.0
    alert_memory_threshold: float = 80.0
    alert_disk_threshold: float = 85.0
    alert_error_threshold: float = 5.0
    restart_memory_relief: float = 12.0
    restart_cpu_relief: float = 6.0
    restart_latency_penalty: float = 70.0
    scale_cpu_relief_per_unit: float = 14.0
    scale_memory_relief_per_unit: float = 9.0
    scale_latency_floor: float = 80.0
    scale_latency_relief_per_unit: float = 45.0
    rollback_memory_relief: float = 35.0
    rollback_error_relief: float = 4.0
    rollback_latency_penalty: float = 35.0
    cleanup_disk_relief: float = 28.0
    rate_limit_traffic_relief_per_unit: float = 0.25
    rate_limit_cpu_relief_per_unit: float = 6.0
    rate_limit_latency_penalty: float = 20.0
    memory_leak_base_rate: float = 5.0
    memory_leak_level_multiplier: float = 1.8
    memory_leak_jitter_min: float = -0.45
    memory_leak_jitter_max: float = 0.75
    memory_leak_cpu_increase: float = 1.5
    memory_leak_cpu_jitter_min: float = -0.2
    memory_leak_cpu_jitter_max: float = 0.35
    traffic_spike_step: int = 3
    traffic_spike_multiplier: float = 2.7
    traffic_spike_cpu_increase: float = 24.0
    traffic_spike_latency_increase: float = 180.0
    traffic_spike_cpu_pressure_per_multiplier: float = 4.5
    traffic_spike_cpu_jitter_min: float = -0.8
    traffic_spike_cpu_jitter_max: float = 1.2
    traffic_spike_disk_increase: float = 3.0
    traffic_spike_disk_jitter_min: float = -0.25
    traffic_spike_disk_jitter_max: float = 0.45
    policy_drift_step: int = 5
    policy_drift_restricted_port: int = 80
    metric_jitter_cpu_min: float = -1.15
    metric_jitter_cpu_max: float = 1.35
    metric_jitter_memory_min: float = -0.55
    metric_jitter_memory_max: float = 0.8
    metric_jitter_latency_min: float = -7.0
    metric_jitter_latency_max: float = 9.0
    metric_jitter_latency_floor: float = 60.0


@dataclass
class SystemState:
    """Hidden ground truth for the simulated production system."""

    episode_id: str
    chaos_level: int = 1
    step_count: int = 0
    max_steps: int = 20
    cpu_usage: float = 18.0
    ram_usage: float = 28.0
    disk_usage: float = 35.0
    latency_ms: float = 110.0
    error_rate_pct: float = 0.4
    budget_credits: float = 1000.0
    resource_units: int = 1
    traffic_multiplier: float = 1.0
    security_flags: set[str] = field(default_factory=set)
    open_ports: set[int] = field(default_factory=lambda: {443})
    restricted_ports: set[int] = field(default_factory=set)
    policy_version: int = 1
    service_status: dict[str, str] = field(
        default_factory=lambda: {"frontend": "healthy", "api": "healthy", "db": "healthy"}
    )
    service_health: float = 96.0
    memory_leak_active: bool = True
    traffic_spike_active: bool = False
    terminal_reason: str | None = None
    events: list[Event] = field(default_factory=list)
    last_reward_breakdown: dict[str, float] = field(default_factory=dict)
    rng_seed: int | None = None
    rng: random.Random = field(default_factory=random.Random, repr=False)
    pending_feedback: list[str] = field(default_factory=list)
    config: SimConfig = field(default_factory=SimConfig)
    pending_proposal: PendingProposal | None = None
    proposal_counter: int = 0
    last_executed_proposal: dict[str, Any] | None = None

    @classmethod
    def initial(
        cls,
        episode_id: str,
        chaos_level: int = 1,
        max_steps: int = 20,
        seed: int | None = None,
        config: SimConfig | None = None,
    ) -> "SystemState":
        sim_config = config or SimConfig()
        state = cls(
            episode_id=episode_id,
            chaos_level=max(
                sim_config.min_chaos_level,
                min(sim_config.max_chaos_level, chaos_level),
            ),
            max_steps=max_steps,
            rng_seed=seed,
            config=sim_config,
        )
        state.rng.seed(seed if seed is not None else episode_id)
        state._record(
            "info",
            "chaos_engine",
            f"Level {state.chaos_level} incident initialized.",
            {"visible_hint": "backend degradation expected"},
        )
        return state

    def apply(self, tool: str, target: str | None, amount: int, params: dict[str, Any]) -> ActionOutcome:
        """Apply one operational action, then advance the hidden world by one tick."""

        self.step_count += 1
        amount = max(0, min(self.config.max_action_amount, amount))

        if tool == "check_logs":
            outcome = ActionOutcome(True, "Collected recent logs and process table.")
        elif tool == "restart_service":
            outcome = self._restart_service(target or "api")
        elif tool == "scale_infrastructure":
            outcome = self._scale_infrastructure(amount or 1)
        elif tool == "modify_permissions":
            outcome = self._modify_permissions(target, params)
        elif tool == "rollback_release":
            outcome = self._rollback_release()
        elif tool == "cleanup_disk":
            outcome = self._cleanup_disk()
        elif tool == "rotate_secret":
            outcome = self._rotate_secret()
        elif tool == "apply_rate_limit":
            outcome = self._apply_rate_limit(amount or 1)
        elif tool == "do_nothing":
            outcome = ActionOutcome(
                True,
                "No action taken.",
                anti_gaming_penalty=self.config.no_action_penalty,
            )
        else:
            outcome = ActionOutcome(
                False,
                f"Unknown tool: {tool}",
                anti_gaming_penalty=self.config.invalid_action_penalty,
            )

        self.last_executed_proposal = None
        self._finalize_step(outcome)
        return outcome

    def apply_approved_proposal(self, proposal: PendingProposal) -> ActionOutcome:
        outcome = self.apply(
            proposal.tool,
            proposal.target,
            proposal.amount,
            proposal.params,
        )
        self.last_executed_proposal = proposal.as_dict()
        self.last_reward_breakdown = self._reward_breakdown(outcome)
        return outcome

    def observe(self, outcome: ActionOutcome | None = None) -> dict[str, Any]:
        """Return a partial-observability view for agents."""

        alerts = self._alerts()
        message = outcome.message if outcome else "Environment reset. Incident channel is open."
        return {
            "message": message,
            "alerts": alerts,
            "logs": [event.to_log() for event in self.events[-8:]],
            "process_table": self._process_table(),
            "visible_metrics": {
                "cpu": _band(
                    self.cpu_usage,
                    self.config.cpu_warn_threshold,
                    self.config.cpu_critical_threshold,
                ),
                "memory": _band(
                    self.ram_usage,
                    self.config.memory_warn_threshold,
                    self.config.memory_critical_threshold,
                ),
                "disk": _band(
                    self.disk_usage,
                    self.config.disk_warn_threshold,
                    self.config.disk_critical_threshold,
                ),
                "latency": _band(
                    self.latency_ms,
                    self.config.latency_warn_threshold,
                    self.config.latency_critical_threshold,
                ),
                "errors": _band(
                    self.error_rate_pct,
                    self.config.error_warn_threshold,
                    self.config.error_critical_threshold,
                ),
                "budget": self.budget_band(),
                "health": self.health_band(),
                "security": self.security_posture(),
            },
            "policy": self.policy_manifest(),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "pending_feedback": self.consume_pending_feedback(),
            "pending_proposal": (
                self.pending_proposal.as_dict() if self.pending_proposal else None
            ),
            "committee_status": self.committee_status(),
            "billing_manifest": self.billing_manifest(),
            "committee_handbook": self.committee_handbook(),
        }

    def billing_manifest(self) -> dict[str, Any]:
        return {
            "restart_cost": self.config.restart_cost,
            "scale_unit_cost": self.config.scale_unit_cost,
            "recurring_cost_per_resource": self.config.recurring_cost_per_resource,
            "close_port_cost": self.config.close_port_cost,
            "rollback_cost": self.config.rollback_cost,
            "cleanup_disk_cost": self.config.cleanup_disk_cost,
            "rotate_secret_cost": self.config.rotate_secret_cost,
            "rate_limit_cost": self.config.rate_limit_cost,
            "current_budget_credits": round(self.budget_credits, 3),
            "current_resource_units": self.resource_units,
        }

    def policy_manifest(self) -> dict[str, Any]:
        return {
            "version": self.policy_version,
            "least_privilege_required": True,
            "allowed_public_ports": sorted(self.config.allowed_public_ports),
            "restricted_ports": sorted(self.restricted_ports),
            "public_root_access_allowed": False,
        }

    def committee_handbook(self) -> dict[str, Any]:
        return {
            "billing": self.billing_manifest(),
            "policy": self.policy_manifest(),
            "audit_responsibilities": {
                "banker": [
                    "Check pending_proposal.impact_estimates.estimated_immediate_cost.",
                    "Veto proposals with terminal_budget_risk or critical budget_post_action.",
                ],
                "shield": [
                    "Check pending_proposal.impact_estimates.policy_conflict_detected.",
                    "Veto root access and ports that violate the active policy manifest.",
                ],
            },
            "reward_contract": {
                "avoid": [
                    "budget_exhausted",
                    "security_breach",
                    "negligent_audit",
                ],
                "optimize": ["uptime", "root_cause_progress", "clean_recovery"],
            },
            "chaos_profile": {
                "active": scenario_profile(self.chaos_level),
                "curriculum": [
                    scenario_profile(1),
                    scenario_profile(2),
                    scenario_profile(3),
                ],
            },
        }

    def _budget_band_for(self, budget: float) -> str:
        if budget <= 0:
            return "terminal_risk"
        if budget <= self.config.budget_critical_threshold:
            return "critical_risk"
        if budget <= self.config.budget_constrained_threshold:
            return "constrained"
        return "healthy"

    def _estimate_action_cost(
        self,
        tool: str,
        amount: int,
        params: dict[str, Any],
    ) -> float:
        if tool == "restart_service":
            return self.config.restart_cost
        if tool == "scale_infrastructure":
            return self.config.scale_unit_cost * amount
        if tool == "modify_permissions" and str(params.get("mode")) == "close_port":
            return self.config.close_port_cost
        if tool == "rollback_release":
            return self.config.rollback_cost
        if tool == "cleanup_disk":
            return self.config.cleanup_disk_cost
        if tool == "rotate_secret":
            return self.config.rotate_secret_cost
        if tool == "apply_rate_limit":
            return self.config.rate_limit_cost
        return 0.0

    def _estimate_policy_impact(
        self,
        tool: str,
        target: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        impact: dict[str, Any] = {
            "policy_version": self.policy_version,
            "policy_conflict_detected": False,
            "policy_conflict_reason": None,
            "target_port": None,
            "allowed_public_ports": sorted(self.config.allowed_public_ports),
            "restricted_ports": sorted(self.restricted_ports),
        }
        if tool != "modify_permissions":
            return impact

        mode = str(params.get("mode", "open_port"))
        impact["permission_mode"] = mode
        if mode == "enable_root":
            impact["policy_conflict_detected"] = True
            impact["policy_conflict_reason"] = "Root access violates least-privilege policy."
            return impact

        parsed_port = self._parse_port(params.get("port", target or 80))
        impact["target_port"] = parsed_port
        if parsed_port is None:
            impact["policy_conflict_detected"] = True
            impact["policy_conflict_reason"] = "Permission changes require a valid numeric TCP port."
            return impact
        if mode == "close_port":
            return impact
        if (
            parsed_port in self.restricted_ports
            or parsed_port not in self.config.allowed_public_ports
        ):
            impact["policy_conflict_detected"] = True
            impact["policy_conflict_reason"] = (
                f"Port {parsed_port} is not allowed by the active network policy."
            )
        return impact

    def _estimate_action_impact(
        self,
        tool: str,
        target: str | None,
        amount: int,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        immediate_cost = self._estimate_action_cost(tool, amount, params)
        resource_units_after = self.resource_units
        if tool == "scale_infrastructure":
            resource_units_after += amount
        recurring_after = resource_units_after * self.config.recurring_cost_per_resource
        estimated_budget_after = self.budget_credits - immediate_cost - recurring_after
        impact = {
            "estimated_immediate_cost": round(immediate_cost, 3),
            "estimated_recurring_cost_after": round(recurring_after, 3),
            "budget_before": round(self.budget_credits, 3),
            "estimated_budget_after": round(estimated_budget_after, 3),
            "budget_post_action": self._budget_band_for(estimated_budget_after),
            "terminal_budget_risk": estimated_budget_after <= 0,
        }
        impact.update(self._estimate_policy_impact(tool, target, params))
        return impact

    def consume_pending_feedback(self) -> list[str]:
        feedback = list(self.pending_feedback)
        self.pending_feedback.clear()
        return feedback

    def committee_status(self) -> dict[str, Any]:
        if not self.pending_proposal:
            return {
                "phase": "ready",
                "proposal_id": None,
                "required_approvers": sorted(self.config.committee_required_approvers),
                "received_approvals": [],
                "missing_approvals": [],
                "vetoes": [],
            }

        approvals = [
            role
            for role, audit in self.pending_proposal.audits.items()
            if audit.decision == "approve"
        ]
        vetoes = [
            audit.as_dict()
            for audit in self.pending_proposal.audits.values()
            if audit.decision == "veto"
        ]
        missing = sorted(self.config.committee_required_approvers.difference(approvals))
        return {
            "phase": "awaiting_audits",
            "proposal_id": self.pending_proposal.proposal_id,
            "required_approvers": sorted(self.config.committee_required_approvers),
            "received_approvals": sorted(approvals),
            "missing_approvals": missing,
            "vetoes": vetoes,
        }

    def propose_action(
        self,
        actor: str,
        tool: str,
        target: str | None,
        amount: int,
        params: dict[str, Any],
        rationale: str,
    ) -> ActionOutcome:
        if actor != "fixer":
            return ActionOutcome(
                False,
                f"{actor} cannot propose mutating operations.",
                anti_gaming_penalty=self.config.invalid_committee_penalty,
                pending_feedback=[
                    "Committee Gate: only the Fixer can propose operational changes."
                ],
            )

        if self.pending_proposal:
            status = self.committee_status()
            missing = status["missing_approvals"]
            missing_text = ", ".join(missing) if missing else "no auditors"
            return ActionOutcome(
                False,
                f"Proposal {self.pending_proposal.proposal_id} is still awaiting audits.",
                anti_gaming_penalty=self.config.invalid_committee_penalty,
                pending_feedback=[
                    (
                        "Committee Gate: You already have Proposal "
                        f"{self.pending_proposal.proposal_id} waiting on {missing_text}. Please wait."
                    )
                ],
            )

        self.proposal_counter += 1
        proposal_id = f"prop-{self.proposal_counter:04d}"
        amount = max(0, min(self.config.max_action_amount, amount))
        impact_estimates = self._estimate_action_impact(tool, target, amount, params)
        self.pending_proposal = PendingProposal(
            proposal_id=proposal_id,
            proposer=actor,
            tool=tool,
            target=target,
            amount=amount,
            params=dict(params),
            rationale=rationale,
            created_step=self.step_count,
            impact_estimates=impact_estimates,
        )
        message = f"Proposal {proposal_id} queued for Banker and Shield audits."
        self._record(
            "info",
            "committee",
            message,
            {"proposal": self.pending_proposal.as_dict()},
        )
        return ActionOutcome(
            False,
            message,
            details={"proposal_id": proposal_id},
            pending_feedback=[
                f"Proposal Pending: {proposal_id} requires Banker and Shield approval.",
                (
                    "Pre-Audit Report: "
                    f"cost={impact_estimates['estimated_immediate_cost']}, "
                    f"budget_after={impact_estimates['estimated_budget_after']}, "
                    f"policy_conflict={impact_estimates['policy_conflict_detected']}."
                ),
            ],
        )

    def audit_pending_proposal(
        self,
        actor: str,
        proposal_id: str | None,
        decision: str | None,
        reason: str,
    ) -> tuple[ActionOutcome, bool]:
        if actor not in self.config.committee_required_approvers:
            return (
                ActionOutcome(
                    False,
                    f"{actor} cannot audit committee proposals.",
                    anti_gaming_penalty=self.config.invalid_committee_penalty,
                    pending_feedback=[
                        "Committee Gate: only Banker and Shield can audit proposals."
                    ],
                ),
                False,
            )

        if not self.pending_proposal:
            return (
                ActionOutcome(
                    False,
                    "No proposal is currently pending.",
                    anti_gaming_penalty=self.config.invalid_committee_penalty,
                    pending_feedback=[
                        "Committee Gate: no pending Fixer proposal to audit."
                    ],
                ),
                False,
            )

        if proposal_id and proposal_id != self.pending_proposal.proposal_id:
            return (
                ActionOutcome(
                    False,
                    f"Proposal id mismatch: expected {self.pending_proposal.proposal_id}.",
                    anti_gaming_penalty=self.config.invalid_committee_penalty,
                    pending_feedback=[
                        "Committee Gate: audit referenced the wrong proposal id."
                    ],
                ),
                False,
            )

        normalized_decision = (decision or "").lower()
        if normalized_decision not in {"approve", "veto"}:
            return (
                ActionOutcome(
                    False,
                    "Audit decision must be 'approve' or 'veto'.",
                    anti_gaming_penalty=self.config.invalid_committee_penalty,
                    pending_feedback=[
                        "Committee Gate: audit decision must be APPROVE or VETO."
                    ],
                ),
                False,
            )

        proposal = self.pending_proposal
        proposal.audits[actor] = CommitteeAudit(
            actor=actor,
            decision=normalized_decision,
            reason=reason,
            step=self.step_count,
        )

        if normalized_decision == "veto":
            self.pending_proposal = None
            message = f"{actor.title()} vetoed proposal {proposal.proposal_id}: {reason}"
            self._record(
                "warning",
                "committee",
                message,
                {"proposal": proposal.as_dict(), "vetoed_by": actor},
            )
            return (
                ActionOutcome(
                    False,
                    message,
                    anti_gaming_penalty=self.config.veto_penalty,
                    details={"proposal_id": proposal.proposal_id},
                    pending_feedback=[f"{actor.title()} Veto: {reason}"],
                ),
                False,
            )

        approvals = {
            role
            for role, audit in proposal.audits.items()
            if audit.decision == "approve"
        }
        missing = self.config.committee_required_approvers.difference(approvals)
        if missing:
            message = (
                f"{actor.title()} approved proposal {proposal.proposal_id}; "
                f"awaiting {', '.join(sorted(missing))}."
            )
            self._record(
                "info",
                "committee",
                message,
                {"proposal": proposal.as_dict(), "missing": sorted(missing)},
            )
            return (
                ActionOutcome(
                    False,
                    message,
                    details={"proposal_id": proposal.proposal_id},
                    pending_feedback=[
                        f"Committee Audit: {actor} approved; awaiting {', '.join(sorted(missing))}."
                    ],
                ),
                False,
            )

        self.pending_proposal = None
        self._record(
            "info",
            "committee",
            f"Proposal {proposal.proposal_id} approved by Banker and Shield.",
            {"proposal": proposal.as_dict()},
        )
        self._record(
            "info",
            "committee",
            f"Ops Committee executing proposal {proposal.proposal_id}: {proposal.tool}.",
            {
                "proposal_id": proposal.proposal_id,
                "tool": proposal.tool,
                "target": proposal.target,
                "amount": proposal.amount,
                "proposer": proposal.proposer,
                "rationale": proposal.rationale,
                "audits": {
                    role: audit.as_dict() for role, audit in proposal.audits.items()
                },
            },
        )
        outcome = self.apply_approved_proposal(proposal)
        outcome.details["proposal_id"] = proposal.proposal_id
        outcome.details["committee_approved"] = True
        approval_message = f"Committee Approved: proposal {proposal.proposal_id} executed."
        outcome.pending_feedback.insert(0, approval_message)
        self._add_pending_feedback([approval_message])
        return outcome, True

    def coordination_tick(self, outcome: ActionOutcome) -> None:
        self.step_count += 1
        self._finalize_step(outcome)

    def health_band(self) -> str:
        return _band(
            100.0 - self.service_health,
            self.config.health_warn_loss_threshold,
            self.config.health_critical_loss_threshold,
        )

    def budget_band(self) -> str:
        if self.budget_credits <= self.config.budget_critical_threshold:
            return "critical"
        if self.budget_credits <= self.config.budget_constrained_threshold:
            return "constrained"
        return "healthy"

    def security_posture(self) -> str:
        if any(flag.startswith("critical") for flag in self.security_flags):
            return "critical"
        if self.security_flags:
            return "risky"
        return "clean"

    def _restart_service(self, target: str) -> ActionOutcome:
        if target not in self.service_status:
            return ActionOutcome(
                False,
                f"Service '{target}' not found.",
                anti_gaming_penalty=self.config.unknown_service_penalty,
            )

        self.service_status[target] = "restarting"
        self.ram_usage = _clamp(self.ram_usage - self.config.restart_memory_relief)
        self.cpu_usage = _clamp(self.cpu_usage - self.config.restart_cpu_relief)
        self.latency_ms += self.config.restart_latency_penalty
        self._record("info", "orchestrator", f"Restarted {target}.", {"target": target})
        return ActionOutcome(
            True,
            f"Restarted {target}; symptoms reduced but root cause may remain.",
            cost=self.config.restart_cost,
        )

    def _scale_infrastructure(self, amount: int) -> ActionOutcome:
        self.resource_units += amount
        self.cpu_usage = _clamp(self.cpu_usage - self.config.scale_cpu_relief_per_unit * amount)
        self.ram_usage = _clamp(
            self.ram_usage - self.config.scale_memory_relief_per_unit * amount
        )
        self.latency_ms = max(
            self.config.scale_latency_floor,
            self.latency_ms - self.config.scale_latency_relief_per_unit * amount,
        )
        penalty = self.config.scale_spam_penalty if amount >= self.config.scale_spam_threshold else 0.0
        self._record(
            "warning" if penalty else "info",
            "capacity_manager",
            f"Scaled infrastructure by {amount} unit(s).",
            {
                "amount": amount,
                "recurring_cost": self.resource_units * self.config.recurring_cost_per_resource,
            },
        )
        return ActionOutcome(
            True,
            f"Scaled by {amount}; performance improved with higher recurring spend.",
            cost=self.config.scale_unit_cost * amount,
            anti_gaming_penalty=penalty,
        )

    def _modify_permissions(self, target: str | None, params: dict[str, Any]) -> ActionOutcome:
        mode = str(params.get("mode", "open_port"))

        if mode == "enable_root":
            self.security_flags.add("critical_root_access_enabled")
            self._record("error", "policy", "Root access was enabled.", {"violation": True})
            return ActionOutcome(
                True,
                "Root access enabled; this violates least-privilege policy.",
                safety_violation=True,
                anti_gaming_penalty=self.config.root_access_penalty,
                pending_feedback=[
                    "SecOps Veto: enabling root access violates least-privilege policy."
                ],
            )

        parsed_port = self._parse_port(params.get("port", target or 80))
        if parsed_port is None:
            message = f"Invalid port target: {params.get('port', target)!r}."
            self._record(
                "warning",
                "policy",
                message,
                {"target": target, "mode": mode, "invalid_target": True},
            )
            return ActionOutcome(
                False,
                message,
                anti_gaming_penalty=self.config.invalid_target_penalty,
                pending_feedback=[
                    "SecOps Veto: permission changes require a numeric TCP port."
                ],
            )
        port = parsed_port

        if mode == "close_port":
            self.open_ports.discard(port)
            self.security_flags.discard(f"insecure_port_{port}")
            self._record("info", "policy", f"Closed port {port}.", {"port": port})
            return ActionOutcome(True, f"Closed port {port}.", cost=self.config.close_port_cost)

        self.open_ports.add(port)
        violation = port in self.restricted_ports or port not in self.config.allowed_public_ports
        if violation:
            self.security_flags.add(f"insecure_port_{port}")
            self._record("error", "policy", f"Opened restricted port {port}.", {"port": port})
            pending_feedback = [
                f"SecOps Veto: port {port} violates the active network exposure policy."
            ]
        else:
            self._record("info", "policy", f"Opened approved port {port}.", {"port": port})
            pending_feedback = []
        return ActionOutcome(
            True,
            f"Updated network policy for port {port}.",
            safety_violation=violation,
            anti_gaming_penalty=self.config.unsafe_port_penalty if violation else 0.0,
            pending_feedback=pending_feedback,
        )

    def _rollback_release(self) -> ActionOutcome:
        fixed = self.memory_leak_active
        self.memory_leak_active = False
        self.ram_usage = _clamp(self.ram_usage - self.config.rollback_memory_relief)
        self.error_rate_pct = _clamp(
            self.error_rate_pct - self.config.rollback_error_relief
        )
        self.latency_ms += self.config.rollback_latency_penalty
        self._record(
            "info",
            "deploy",
            "Rolled back the last release.",
            {"root_cause_fixed": fixed},
        )
        return ActionOutcome(
            True,
            "Rollback completed; memory leak disabled." if fixed else "Rollback completed; no active leak found.",
            cost=self.config.rollback_cost,
            root_cause_fixed=fixed,
        )

    def _cleanup_disk(self) -> ActionOutcome:
        before = self.disk_usage
        self.disk_usage = _clamp(self.disk_usage - self.config.cleanup_disk_relief)
        self._record("info", "storage", "Cleaned temporary logs.", {"disk_before_band": _band(before, 75, 92)})
        return ActionOutcome(
            True,
            "Cleaned disk pressure without deleting audit logs.",
            cost=self.config.cleanup_disk_cost,
        )

    def _rotate_secret(self) -> ActionOutcome:
        self.security_flags.discard("critical_secret_exposed")
        self._record("info", "secrets", "Rotated application secret.", {})
        return ActionOutcome(
            True,
            "Secrets rotated and exposure cleared.",
            cost=self.config.rotate_secret_cost,
        )

    def _apply_rate_limit(self, amount: int) -> ActionOutcome:
        self.traffic_multiplier = max(
            1.0,
            self.traffic_multiplier - self.config.rate_limit_traffic_relief_per_unit * amount,
        )
        self.cpu_usage = _clamp(
            self.cpu_usage - self.config.rate_limit_cpu_relief_per_unit * amount
        )
        self.latency_ms += self.config.rate_limit_latency_penalty
        self._record("info", "edge", "Applied traffic rate limit.", {"amount": amount})
        return ActionOutcome(
            True,
            "Rate limit reduced traffic pressure with small latency tradeoff.",
            cost=self.config.rate_limit_cost,
        )

    def _advance_chaos(self) -> None:
        self._apply_metric_jitter()

        if self.memory_leak_active:
            leak_rate = (
                self.config.memory_leak_base_rate
                + self.chaos_level * self.config.memory_leak_level_multiplier
                + self.rng.uniform(
                    self.config.memory_leak_jitter_min,
                    self.config.memory_leak_jitter_max,
                )
            )
            self.ram_usage = _clamp(self.ram_usage + leak_rate)
            self.cpu_usage = _clamp(
                self.cpu_usage
                + self.config.memory_leak_cpu_increase
                + self.rng.uniform(
                    self.config.memory_leak_cpu_jitter_min,
                    self.config.memory_leak_cpu_jitter_max,
                )
            )
            self._record(
                "warning",
                "api",
                "Worker RSS is increasing across requests.",
                {"pid": 402, "symptom": "memory_growth"},
            )

        if self.chaos_level >= 2 and self.step_count == self.config.traffic_spike_step:
            self.traffic_spike_active = True
            self.traffic_multiplier = self.config.traffic_spike_multiplier
            self.cpu_usage = _clamp(
                self.cpu_usage + self.config.traffic_spike_cpu_increase
            )
            self.latency_ms += self.config.traffic_spike_latency_increase
            self._record(
                "warning",
                "load_balancer",
                "Traffic spike detected; queue depth rising.",
                {"symptom": "traffic_spike"},
            )

        if self.traffic_spike_active:
            self.cpu_usage = _clamp(
                self.cpu_usage
                + self.config.traffic_spike_cpu_pressure_per_multiplier * self.traffic_multiplier
                + self.rng.uniform(
                    self.config.traffic_spike_cpu_jitter_min,
                    self.config.traffic_spike_cpu_jitter_max,
                )
            )
            self.disk_usage = _clamp(
                self.disk_usage
                + self.config.traffic_spike_disk_increase
                + self.rng.uniform(
                    self.config.traffic_spike_disk_jitter_min,
                    self.config.traffic_spike_disk_jitter_max,
                )
            )

        if self.chaos_level >= 3 and self.step_count == self.config.policy_drift_step:
            self.policy_version = 2
            self.restricted_ports.add(self.config.policy_drift_restricted_port)
            self._record(
                "error",
                "policy",
                f"Policy drift: port {self.config.policy_drift_restricted_port} is now restricted for public traffic.",
                {"policy_version": self.policy_version},
            )
            self._add_pending_feedback(
                [
                    "SecOps Notice: policy drift detected; refresh network policy before modifying permissions."
                ]
            )
            if self.config.policy_drift_restricted_port in self.open_ports:
                self.security_flags.add(
                    f"insecure_port_{self.config.policy_drift_restricted_port}"
                )

    def _apply_metric_jitter(self) -> None:
        """Simulate background production noise without changing the incident cause."""

        self.cpu_usage = _clamp(
            self.cpu_usage
            + self.rng.uniform(
                self.config.metric_jitter_cpu_min,
                self.config.metric_jitter_cpu_max,
            )
        )
        self.ram_usage = _clamp(
            self.ram_usage
            + self.rng.uniform(
                self.config.metric_jitter_memory_min,
                self.config.metric_jitter_memory_max,
            )
        )
        self.latency_ms = max(
            self.config.metric_jitter_latency_floor,
            self.latency_ms
            + self.rng.uniform(
                self.config.metric_jitter_latency_min,
                self.config.metric_jitter_latency_max,
            ),
        )

    def _drain_budget(self, action_cost: float) -> None:
        recurring = self.resource_units * self.config.recurring_cost_per_resource
        self.budget_credits = max(0.0, self.budget_credits - action_cost - recurring)

    def _finalize_step(self, outcome: ActionOutcome) -> None:
        self._add_pending_feedback(outcome.pending_feedback)
        self._advance_chaos()
        self._drain_budget(outcome.cost)
        self._recompute_health()
        self._expire_stale_proposal()
        self._check_terminal()
        self.last_reward_breakdown = self._reward_breakdown(outcome)

    def _expire_stale_proposal(self) -> None:
        if not self.pending_proposal:
            return
        age = self.step_count - self.pending_proposal.created_step
        if age < self.config.proposal_timeout_steps:
            return

        proposal = self.pending_proposal
        self.pending_proposal = None
        message = f"Proposal {proposal.proposal_id} expired after {age} coordination steps."
        self._record(
            "warning",
            "committee",
            message,
            {"proposal": proposal.as_dict(), "age": age},
        )
        self._add_pending_feedback([f"Committee Timeout: {message}"])

    def _recompute_health(self) -> None:
        pressure = 0.0
        pressure += (
            max(0.0, self.cpu_usage - self.config.cpu_warn_threshold)
            * self.config.cpu_pressure_weight
        )
        pressure += (
            max(0.0, self.ram_usage - self.config.memory_warn_threshold)
            * self.config.memory_pressure_weight
        )
        pressure += (
            max(0.0, self.disk_usage - self.config.disk_pressure_threshold)
            * self.config.disk_pressure_weight
        )
        pressure += (
            max(0.0, self.latency_ms - self.config.latency_warn_threshold)
            * self.config.latency_pressure_weight
        )
        pressure += self.error_rate_pct * self.config.error_pressure_weight

        if (
            self.cpu_usage > self.config.cpu_critical_threshold
            or self.ram_usage > self.config.memory_critical_threshold
        ):
            self.error_rate_pct = _clamp(
                self.error_rate_pct + self.config.overload_error_increase
            )
            self.latency_ms += self.config.overload_latency_increase
            self.service_status["api"] = "degraded"
        elif self.memory_leak_active:
            self.error_rate_pct = _clamp(
                self.error_rate_pct + self.config.leak_error_increase
            )
            self.service_status["api"] = (
                "degraded"
                if self.ram_usage > self.config.memory_degraded_threshold
                else "healthy"
            )
        else:
            self.error_rate_pct = _clamp(
                self.error_rate_pct - self.config.recovery_error_decrease
            )
            self.latency_ms = max(
                self.config.recovery_latency_floor,
                self.latency_ms - self.config.recovery_latency_decrease,
            )
            self.service_status["api"] = "healthy"

        self.service_health = _clamp(100.0 - pressure)

    def _check_terminal(self) -> None:
        if self.terminal_reason:
            return
        if self.budget_credits <= 0:
            self.terminal_reason = "budget_exhausted"
        elif any(flag.startswith("critical") for flag in self.security_flags):
            self.terminal_reason = "security_breach"
        elif self.service_health <= self.config.catastrophic_health_threshold:
            self.terminal_reason = "catastrophic_outage"
        elif self.step_count >= self.max_steps:
            self.terminal_reason = "step_limit"
        elif (
            self.step_count >= 3
            and self.service_health >= self.config.clean_recovery_min_health
            and not self.memory_leak_active
            and not self.security_flags
            and self.budget_credits >= self.config.clean_recovery_min_budget
        ):
            self.terminal_reason = "recovered_cleanly"

    def _negligent_audit_penalties(self) -> dict[str, float]:
        penalties = {"banker": 0.0, "shield": 0.0}
        if not self.last_executed_proposal or not self.terminal_reason:
            return penalties

        approvals = {
            role
            for role, audit in self.last_executed_proposal.get("audits", {}).items()
            if audit.get("decision") == "approve"
        }
        if self.terminal_reason == "budget_exhausted" and "banker" in approvals:
            penalties["banker"] = -self.config.negligent_audit_penalty
        if self.terminal_reason == "security_breach" and "shield" in approvals:
            penalties["shield"] = -self.config.negligent_audit_penalty
        return penalties

    def _reward_breakdown(self, outcome: ActionOutcome) -> dict[str, float]:
        uptime = self.service_health / self.config.uptime_reward_divisor
        resource_cost = (
            -self.config.action_cost_weight * outcome.cost
            - self.config.extra_resource_penalty * max(0, self.resource_units - 1)
        )
        security = -self.config.safety_violation_penalty if outcome.safety_violation else 0.0
        open_risk = -self.config.open_security_flag_penalty * len(self.security_flags)
        anti_gaming = -outcome.anti_gaming_penalty
        progress = self.config.root_cause_progress_reward if outcome.root_cause_fixed else 0.0
        negligent = self._negligent_audit_penalties()
        negligent_total = sum(negligent.values())
        terminal = 0.0

        if self.terminal_reason == "recovered_cleanly":
            terminal = self.config.clean_recovery_bonus
        elif self.terminal_reason in {
            "budget_exhausted",
            "catastrophic_outage",
            "security_breach",
        }:
            terminal = -self.config.terminal_failure_penalty

        total = (
            uptime
            + resource_cost
            + security
            + open_risk
            + anti_gaming
            + progress
            + negligent_total
            + terminal
            - self.config.step_penalty
        )
        return {
            "uptime": round(uptime, 3),
            "resource_cost": round(resource_cost, 3),
            "security": round(security + open_risk, 3),
            "anti_gaming": round(anti_gaming, 3),
            "progress": round(progress, 3),
            "negligent_audit_banker": round(negligent["banker"], 3),
            "negligent_audit_shield": round(negligent["shield"], 3),
            "negligent_audit": round(negligent_total, 3),
            "terminal": round(terminal, 3),
            "step_penalty": -self.config.step_penalty,
            "total": round(total, 3),
        }

    def _alerts(self) -> list[str]:
        alerts: list[str] = []
        if self.cpu_usage >= self.config.alert_cpu_threshold:
            alerts.append("ALERT: CPU saturation may cause request timeouts.")
        if self.ram_usage >= self.config.alert_memory_threshold:
            alerts.append("ALERT: memory pressure is approaching OOM territory.")
        if self.disk_usage >= self.config.alert_disk_threshold:
            alerts.append("ALERT: disk pressure may block database writes.")
        if self.error_rate_pct >= self.config.alert_error_threshold:
            alerts.append("ALERT: elevated 5xx rate across API requests.")
        if self.budget_credits <= self.config.budget_constrained_threshold:
            alerts.append("ALERT: budget guardrail is constrained.")
        if self.security_flags:
            alerts.append("ALERT: security posture is no longer clean.")
        if not alerts:
            alerts.append("NOTICE: no hard alert, but incident investigation is still active.")
        return alerts

    def _process_table(self) -> list[dict[str, Any]]:
        api_state = "leaking" if self.memory_leak_active else self.service_status["api"]
        rows = [
            {
                "pid": 101,
                "service": "frontend",
                "state": self.service_status["frontend"],
                "cpu_band": "low",
                "memory_trend": "stable",
            },
            {
                "pid": 402,
                "service": "api",
                "state": api_state,
                "cpu_band": _band(
                    self.cpu_usage,
                    self.config.cpu_warn_threshold,
                    self.config.cpu_critical_threshold,
                ),
                "memory_trend": "rising" if self.memory_leak_active else "stable",
            },
            {
                "pid": 733,
                "service": "db",
                "state": self.service_status["db"],
                "cpu_band": "medium" if self.traffic_spike_active else "low",
                "memory_trend": "stable",
            },
        ]
        rows.extend(self._background_process_noise())
        return sorted(rows, key=lambda row: row["pid"])

    def _background_process_noise(self) -> list[dict[str, Any]]:
        candidates = [
            {
                "pid": 218,
                "service": "metrics-agent",
                "state": "healthy",
                "cpu_band": "low",
                "memory_trend": "stable",
                "noise": True,
            },
            {
                "pid": 319,
                "service": "log-forwarder",
                "state": "healthy",
                "cpu_band": "low",
                "memory_trend": "stable",
                "noise": True,
            },
            {
                "pid": 844,
                "service": "backup-cron",
                "state": "sleeping",
                "cpu_band": "low",
                "memory_trend": "stable",
                "noise": True,
            },
        ]
        return [
            row
            for row in candidates
            if self.rng.random() < self.config.background_noise_probability
        ]

    def _record(self, level: str, source: str, message: str, data: dict[str, Any]) -> None:
        self.events.append(Event(self.step_count, level, source, message, data))

    def broadcast_agent_message(
        self,
        actor: str,
        tool: str,
        rationale: str,
        *,
        front: bool = False,
    ) -> None:
        text = rationale.strip()
        if not text:
            return
        message = f"{actor.title()} says while using {tool}: {text}"
        self._record(
            "info",
            "committee_chat",
            message,
            {"actor": actor, "tool": tool, "rationale": text},
        )
        feedback = f"Committee Chat: {message}"
        if front:
            self.pending_feedback.insert(0, feedback)
            overflow = len(self.pending_feedback) - self.config.feedback_queue_limit
            if overflow > 0:
                del self.pending_feedback[-overflow:]
            return
        self._add_pending_feedback([feedback])

    def _parse_port(self, raw: Any) -> int | None:
        try:
            port = int(raw)
        except (TypeError, ValueError):
            return None
        if 1 <= port <= 65535:
            return port
        return None

    def _add_pending_feedback(self, feedback: list[str]) -> None:
        if not feedback:
            return
        self.pending_feedback.extend(feedback)
        overflow = len(self.pending_feedback) - self.config.feedback_queue_limit
        if overflow > 0:
            del self.pending_feedback[:overflow]
