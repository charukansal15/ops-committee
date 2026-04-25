"""FastAPI entry point for the Ops Committee OpenEnv server."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "OpenEnv is required to run the web server. Install project dependencies first."
    ) from exc

from ops_committee_env.models import OpsCommitteeAction, OpsCommitteeObservation
from ops_committee_env.server.ops_committee_environment import OpsCommitteeEnvironment


app = create_app(
    OpsCommitteeEnvironment,
    OpsCommitteeAction,
    OpsCommitteeObservation,
    env_name="ops_committee_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

