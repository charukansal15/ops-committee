"""OpenEnv wrapper around the Ops Committee deterministic simulator."""

from __future__ import annotations

from uuid import uuid4

from ops_committee_env.compat import Environment
from ops_committee_env.models import (
    AgentRole,
    AuditDecision,
    OpsCommitteeAction,
    OpsCommitteeObservation,
    OpsCommitteeState,
    OpsTool,
)
from ops_committee_env.server.system_state import ActionOutcome, SystemState
from ops_committee_env.server.tools import (
    get_tool_spec,
    list_tool_specs,
    role_value,
    validate_tool_names,
)


validate_tool_names()


class OpsCommitteeEnvironment(
    Environment[OpsCommitteeAction, OpsCommitteeObservation, OpsCommitteeState]
):
    """Stateful incident-response environment with partial observability."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._world = SystemState.initial(episode_id=str(uuid4()), chaos_level=1)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        level: int = 1,
        max_steps: int = 20,
        **_: object,
    ) -> OpsCommitteeObservation:
        self._world = SystemState.initial(
            episode_id=episode_id or str(uuid4()),
            chaos_level=level,
            max_steps=max_steps,
            seed=seed,
        )
        observation = self._make_observation()
        observation.reward = 0.0
        return observation

    def step(  # type: ignore[override]
        self,
        action: OpsCommitteeAction | dict,
        timeout_s: float | None = None,
        **_: object,
    ) -> OpsCommitteeObservation:
        del timeout_s
        if isinstance(action, dict):
            action = OpsCommitteeAction(**action)

        outcome = self._route_action(action)
        observation = self._make_observation(outcome)
        observation.reward = self._world.last_reward_breakdown["total"]
        observation.done = self._world.terminal_reason is not None
        return observation

    @property
    def state(self) -> OpsCommitteeState:
        return OpsCommitteeState(
            episode_id=self._world.episode_id,
            step_count=self._world.step_count,
            incident_level=self._world.chaos_level,
            chaos_level=self._world.chaos_level,
            policy_version=self._world.policy_version,
            health_band=self._world.health_band(),
            budget_band=self._world.budget_band(),
            security_posture=self._world.security_posture(),
            committee_phase=self._world.committee_status()["phase"],
            pending_proposal_id=(
                self._world.pending_proposal.proposal_id
                if self._world.pending_proposal
                else None
            ),
            terminal_reason=self._world.terminal_reason,
        )

    def list_tools(self) -> list[dict]:
        return list_tool_specs(self._world.config)

    def check_logs(self, actor: AgentRole = AgentRole.FIXER) -> OpsCommitteeObservation:
        return self.step(OpsCommitteeAction(actor=actor, tool=OpsTool.CHECK_LOGS))

    def restart_service(
        self,
        service: str = "api",
        rationale: str = "",
    ) -> OpsCommitteeObservation:
        return self.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.RESTART_SERVICE,
                target=service,
                rationale=rationale,
            )
        )

    def scale_infrastructure(
        self,
        amount: int,
        rationale: str = "",
    ) -> OpsCommitteeObservation:
        return self.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.SCALE_INFRASTRUCTURE,
                amount=amount,
                rationale=rationale,
            )
        )

    def modify_permissions(
        self,
        mode: str,
        port: int | None = None,
        rationale: str = "",
    ) -> OpsCommitteeObservation:
        params = {"mode": mode}
        if port is not None:
            params["port"] = port
        return self.step(
            OpsCommitteeAction(
                actor=AgentRole.FIXER,
                tool=OpsTool.MODIFY_PERMISSIONS,
                target=str(port) if port is not None else None,
                params=params,
                rationale=rationale,
            )
        )

    def audit_proposal(
        self,
        actor: AgentRole,
        decision: AuditDecision,
        proposal_id: str | None = None,
        rationale: str = "",
    ) -> OpsCommitteeObservation:
        return self.step(
            OpsCommitteeAction(
                actor=actor,
                tool=OpsTool.AUDIT_PROPOSAL,
                proposal_id=proposal_id,
                decision=decision,
                rationale=rationale,
            )
        )

    def _route_action(self, action: OpsCommitteeAction) -> ActionOutcome:
        tool = str(getattr(action.tool, "value", action.tool))
        actor = str(role_value(action.actor))
        spec = get_tool_spec(tool)
        chat_after_gate = tool == OpsTool.AUDIT_PROPOSAL.value
        if not chat_after_gate:
            self._world.broadcast_agent_message(actor, tool, action.rationale)

        if spec and actor not in spec.allowed_roles:
            outcome = ActionOutcome(
                False,
                f"{actor} is not allowed to call {tool}.",
                anti_gaming_penalty=self._world.config.invalid_committee_penalty,
                pending_feedback=[
                    f"Committee Gate: {tool} is allowed for {', '.join(spec.allowed_roles)}."
                ],
            )
            self._world.coordination_tick(outcome)
            return outcome

        if tool == OpsTool.AUDIT_PROPOSAL.value:
            decision = getattr(action.decision, "value", action.decision)
            outcome, applied = self._world.audit_pending_proposal(
                actor=actor,
                proposal_id=action.proposal_id,
                decision=str(decision) if decision is not None else None,
                reason=action.rationale or "No audit rationale provided.",
            )
            if not applied:
                self._world.coordination_tick(outcome)
            self._world.broadcast_agent_message(
                actor,
                tool,
                action.rationale,
                front=applied,
            )
            return outcome

        if spec and spec.requires_approval:
            outcome = self._world.propose_action(
                actor=actor,
                tool=tool,
                target=action.target,
                amount=action.amount,
                params=action.params,
                rationale=action.rationale,
            )
            self._world.coordination_tick(outcome)
            return outcome

        return self._world.apply(
            tool=tool,
            target=action.target,
            amount=action.amount,
            params=action.params,
        )

    def _make_observation(
        self,
        outcome: ActionOutcome | None = None,
    ) -> OpsCommitteeObservation:
        payload = self._world.observe(outcome)
        return OpsCommitteeObservation(
            message=payload["message"],
            alerts=payload["alerts"],
            logs=payload["logs"],
            process_table=payload["process_table"],
            visible_metrics=payload["visible_metrics"],
            policy=payload["policy"],
            reward_breakdown=payload["reward_breakdown"],
            pending_feedback=payload["pending_feedback"],
            pending_proposal=payload["pending_proposal"],
            committee_status=payload["committee_status"],
            registered_tools=list_tool_specs(self._world.config),
            billing_manifest=payload["billing_manifest"],
            committee_handbook=payload["committee_handbook"],
            done=self._world.terminal_reason is not None,
            reward=self._world.last_reward_breakdown.get("total"),
            metadata={
                "episode_id": self._world.episode_id,
                "step_count": self._world.step_count,
            },
        )
