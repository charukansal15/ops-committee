"""Typed action, observation, and public state models for The Ops Committee."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .compat import Action, Field, Observation, State


class AgentRole(str, Enum):
    FIXER = "fixer"
    BANKER = "banker"
    SHIELD = "shield"
    SYSTEM = "system"


class OpsTool(str, Enum):
    CHECK_LOGS = "check_logs"
    RESTART_SERVICE = "restart_service"
    SCALE_INFRASTRUCTURE = "scale_infrastructure"
    MODIFY_PERMISSIONS = "modify_permissions"
    ROLLBACK_RELEASE = "rollback_release"
    CLEANUP_DISK = "cleanup_disk"
    ROTATE_SECRET = "rotate_secret"
    APPLY_RATE_LIMIT = "apply_rate_limit"
    DO_NOTHING = "do_nothing"
    AUDIT_PROPOSAL = "audit_proposal"


class AuditDecision(str, Enum):
    APPROVE = "approve"
    VETO = "veto"


class OpsCommitteeAction(Action):
    """Agent action submitted to the operational sandbox."""

    actor: AgentRole = Field(
        default=AgentRole.FIXER,
        description="Committee member proposing the action.",
    )
    tool: OpsTool = Field(description="Operational tool to execute.")
    target: str | None = Field(
        default=None,
        description="Service, port, file, or policy target for the tool.",
    )
    amount: int = Field(
        default=1,
        ge=0,
        description="Small integer magnitude for resource-changing tools.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific parameters.",
    )
    proposal_id: str | None = Field(
        default=None,
        description="Committee proposal identifier for audit actions.",
    )
    decision: AuditDecision | None = Field(
        default=None,
        description="Banker or Shield decision for a pending proposal.",
    )
    rationale: str = Field(
        default="",
        description="Short reason supplied by the acting agent.",
    )


class OpsCommitteeObservation(Observation):
    """Partially observable incident response view returned to agents."""

    message: str = Field(default="", description="Human-readable action result.")
    alerts: list[str] = Field(default_factory=list, description="Active alerts.")
    logs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent structured log lines.",
    )
    process_table: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Partial process table visible to agents.",
    )
    visible_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Coarse metric bands instead of hidden raw values.",
    )
    policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Currently visible operating policy.",
    )
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Composable reward ledger for the latest step.",
    )
    pending_feedback: list[str] = Field(
        default_factory=list,
        description="Coordination or veto messages for later phases.",
    )
    pending_proposal: dict[str, Any] | None = Field(
        default=None,
        description="Current Fixer proposal awaiting Banker and Shield audits.",
    )
    committee_status: dict[str, Any] = Field(
        default_factory=dict,
        description="Negotiation phase, received audits, and missing approvers.",
    )
    registered_tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent-facing tool manifest for OpenEnv/MCP mapping.",
    )
    billing_manifest: dict[str, Any] = Field(
        default_factory=dict,
        description="Read-only cost ledger visible to FinOps auditors.",
    )
    committee_handbook: dict[str, Any] = Field(
        default_factory=dict,
        description="Read-only audit responsibilities and policy/rubric contract.",
    )


class OpsCommitteeState(State):
    """Public environment state, intentionally excluding raw ground truth."""

    incident_level: int = Field(default=1)
    chaos_level: int = Field(default=1)
    policy_version: int = Field(default=1)
    health_band: str = Field(default="nominal")
    budget_band: str = Field(default="healthy")
    security_posture: str = Field(default="clean")
    committee_phase: str = Field(default="ready")
    pending_proposal_id: str | None = Field(default=None)
    terminal_reason: str | None = Field(default=None)
