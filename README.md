# The Ops Committee

The Ops Committee is an OpenEnv-compatible incident-response environment for
training agents to balance uptime, cost, and security under adversarial faults.

## Winning Blueprint

The environment simulates a generic backend deployment rather than a single
cloud provider. A hidden chaos engine degrades the system, while a committee of
agents must negotiate safe remediation.

Primary themes:

- Theme 1, Multi-Agent: Fixer proposes actions while Banker and Shield audit
  cost and security risk.
- Theme 2, Long-Horizon: early shortcuts create delayed outages, budget drains,
  or policy violations.
- Theme 3, World Modeling: agents reason from logs, alerts, process tables, and
  policy messages instead of raw ground-truth variables.
- Theme 4, Self-Improvement: later phases escalate chaos difficulty based on
  performance history.

Phase 1 status:

- Deterministic Python state machine.
- Hidden ground-truth system metrics.
- Partial-observability observation generator.
- OpenEnv-shaped `reset()`, `step()`, and `state` wrapper.
- Initial reward ledger for uptime, spend, safety, and anti-gaming penalties.

Phase 2 status:

- Agent-facing tool registry with reserved-name validation.
- Mutating Fixer tools are routed into pending proposals.
- Banker and Shield audit proposals with `approve` or `veto` decisions.
- Approved proposals execute only after both audits are present.
- Vetoed proposals are cleared and returned as structured feedback.
- Observations include `pending_proposal`, `committee_status`, and
  `registered_tools`.

## Local Smoke Test

The core simulator can run without OpenEnv installed:

```powershell
python -m unittest discover tests
```

Once OpenEnv dependencies are installed, the server entry point is:

```powershell
uvicorn ops_committee_env.server.app:app --host 0.0.0.0 --port 8000
```
