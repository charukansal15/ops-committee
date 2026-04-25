"""Agent-facing tool registry for The Ops Committee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ops_committee_env.models import AgentRole, OpsTool


RESERVED_TOOL_NAMES = {"reset", "step", "state", "close"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    allowed_roles: tuple[str, ...]
    mutating: bool
    requires_approval: bool
    parameters: dict[str, Any]

    def as_dict(self, config: Any | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_roles": list(self.allowed_roles),
            "mutating": self.mutating,
            "requires_approval": self.requires_approval,
            "parameters": self.parameters,
            "cost_model": _cost_model_for_tool(self.name, config),
        }


ALL_ROLES = tuple(role.value for role in AgentRole if role is not AgentRole.SYSTEM)
FIXER_ONLY = (AgentRole.FIXER.value,)
AUDITORS = (AgentRole.BANKER.value, AgentRole.SHIELD.value)


TOOL_SPECS: dict[str, ToolSpec] = {
    OpsTool.CHECK_LOGS.value: ToolSpec(
        name=OpsTool.CHECK_LOGS.value,
        description="Inspect partial logs, process table, alerts, policy, and metric bands.",
        allowed_roles=ALL_ROLES,
        mutating=False,
        requires_approval=False,
        parameters={},
    ),
    OpsTool.RESTART_SERVICE.value: ToolSpec(
        name=OpsTool.RESTART_SERVICE.value,
        description="Propose restarting a named service such as frontend, api, or db.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={"target": "service name"},
    ),
    OpsTool.SCALE_INFRASTRUCTURE.value: ToolSpec(
        name=OpsTool.SCALE_INFRASTRUCTURE.value,
        description="Propose adding resource units to reduce pressure at a recurring cost.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={"amount": "integer resource units"},
    ),
    OpsTool.MODIFY_PERMISSIONS.value: ToolSpec(
        name=OpsTool.MODIFY_PERMISSIONS.value,
        description="Propose opening, closing, or changing a permission boundary.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={"mode": "open_port | close_port | enable_root", "port": "TCP port"},
    ),
    OpsTool.ROLLBACK_RELEASE.value: ToolSpec(
        name=OpsTool.ROLLBACK_RELEASE.value,
        description="Propose rolling back the latest application release.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={},
    ),
    OpsTool.CLEANUP_DISK.value: ToolSpec(
        name=OpsTool.CLEANUP_DISK.value,
        description="Propose cleaning temporary data without deleting audit logs.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={},
    ),
    OpsTool.ROTATE_SECRET.value: ToolSpec(
        name=OpsTool.ROTATE_SECRET.value,
        description="Propose rotating a leaked application secret.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={},
    ),
    OpsTool.APPLY_RATE_LIMIT.value: ToolSpec(
        name=OpsTool.APPLY_RATE_LIMIT.value,
        description="Propose adding rate limits to trade throughput for stability.",
        allowed_roles=FIXER_ONLY,
        mutating=True,
        requires_approval=True,
        parameters={"amount": "integer rate-limit strength"},
    ),
    OpsTool.DO_NOTHING.value: ToolSpec(
        name=OpsTool.DO_NOTHING.value,
        description="Spend one step without changing the system.",
        allowed_roles=ALL_ROLES,
        mutating=False,
        requires_approval=False,
        parameters={},
    ),
    OpsTool.AUDIT_PROPOSAL.value: ToolSpec(
        name=OpsTool.AUDIT_PROPOSAL.value,
        description="Approve or veto the current Fixer proposal as Banker or Shield.",
        allowed_roles=AUDITORS,
        mutating=False,
        requires_approval=False,
        parameters={"proposal_id": "proposal id", "decision": "approve | veto"},
    ),
}


def list_tool_specs(config: Any | None = None) -> list[dict[str, Any]]:
    return [spec.as_dict(config) for spec in TOOL_SPECS.values()]


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(tool_name)


def validate_tool_names() -> None:
    reserved = RESERVED_TOOL_NAMES.intersection(TOOL_SPECS)
    if reserved:
        raise ValueError(f"Tool registry uses reserved names: {sorted(reserved)}")


def role_value(role: Any) -> str:
    return getattr(role, "value", role)


def _cfg(config: Any | None, name: str, default: float | int | list[int] | None = None) -> Any:
    return getattr(config, name, default)


def _cost_model_for_tool(tool_name: str, config: Any | None) -> dict[str, Any]:
    zero = {
        "estimated_immediate_cost": 0.0,
        "estimated_recurring_cost_after": "unchanged",
        "cost_formula": "0",
    }
    if tool_name == OpsTool.SCALE_INFRASTRUCTURE.value:
        return {
            "cost_per_unit": _cfg(config, "scale_unit_cost", 28.0),
            "recurring_cost_per_resource": _cfg(config, "recurring_cost_per_resource", 2.0),
            "cost_formula": "scale_unit_cost * amount",
        }
    if tool_name == OpsTool.RESTART_SERVICE.value:
        return {"fixed_cost": _cfg(config, "restart_cost", 4.0), "cost_formula": "restart_cost"}
    if tool_name == OpsTool.MODIFY_PERMISSIONS.value:
        return {
            "close_port_cost": _cfg(config, "close_port_cost", 1.0),
            "open_port_cost": 0.0,
            "allowed_public_ports": sorted(_cfg(config, "allowed_public_ports", [])),
            "cost_formula": "close_port_cost only for close_port; opening risky ports may trigger security penalties",
        }
    if tool_name == OpsTool.ROLLBACK_RELEASE.value:
        return {"fixed_cost": _cfg(config, "rollback_cost", 16.0), "cost_formula": "rollback_cost"}
    if tool_name == OpsTool.CLEANUP_DISK.value:
        return {"fixed_cost": _cfg(config, "cleanup_disk_cost", 3.0), "cost_formula": "cleanup_disk_cost"}
    if tool_name == OpsTool.ROTATE_SECRET.value:
        return {"fixed_cost": _cfg(config, "rotate_secret_cost", 5.0), "cost_formula": "rotate_secret_cost"}
    if tool_name == OpsTool.APPLY_RATE_LIMIT.value:
        return {"fixed_cost": _cfg(config, "rate_limit_cost", 2.0), "cost_formula": "rate_limit_cost"}
    return zero
