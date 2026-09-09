"""Public command result contracts; API schemas will be added with the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class CheckResult(BaseModel):
    """A configuration check does not claim model or database readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok"] = "ok"
    scope: Literal["configuration"] = "configuration"
    package_version: str
