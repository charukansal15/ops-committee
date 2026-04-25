"""Small compatibility layer for local development without OpenEnv installed."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

try:  # pragma: no cover - exercised once OpenEnv is installed.
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
    from pydantic import Field

    OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback is covered by tests.
    OPENENV_AVAILABLE = False

    class _FieldValue:
        def __init__(
            self,
            default: Any = None,
            default_factory: Callable[[], Any] | None = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory

        def resolve(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(  # type: ignore[override]
        default: Any = None,
        *,
        default_factory: Callable[[], Any] | None = None,
        description: str | None = None,
        **_: Any,
    ) -> Any:
        del description
        return _FieldValue(default=default, default_factory=default_factory)

    def _dump_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [_dump_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _dump_value(item) for key, item in value.items()}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value

    class _SimpleModel:
        def __init__(self, **kwargs: Any) -> None:
            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))

            for name in annotations:
                if name in kwargs:
                    setattr(self, name, kwargs.pop(name))
                    continue

                default = getattr(self.__class__, name, None)
                if isinstance(default, _FieldValue):
                    default = default.resolve()
                setattr(self, name, default)

            for name, value in kwargs.items():
                setattr(self, name, value)

        def model_dump(self) -> dict[str, Any]:
            return {
                key: _dump_value(value)
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }

        def __repr__(self) -> str:
            fields = ", ".join(
                f"{key}={value!r}" for key, value in self.__dict__.items()
            )
            return f"{self.__class__.__name__}({fields})"

    class Action(_SimpleModel):
        metadata: dict[str, Any] = Field(default_factory=dict)

    class Observation(_SimpleModel):
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(_SimpleModel):
        episode_id: str | None = Field(default=None)
        step_count: int = Field(default=0)

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __class_getitem__(cls, _: Any) -> type["Environment"]:
            return cls

        def __init__(self, *_: Any, **__: Any) -> None:
            pass
