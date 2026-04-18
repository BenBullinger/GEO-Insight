"""Property + Lens + Mode registry, loaded from analysis/spec.yaml.

Every analytic view reads from this registry so that column names surface with
their formal provenance (formula, source, inputs, unit, failure modes). This
is the "Foundry ontology" layer — every aggregation has a declared type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Property:
    name: str
    level: int
    description: str | None = None
    formula: str | None = None
    source: str | None = None
    inputs: list[str] = field(default_factory=list)
    unit: str | None = None
    granularity: str | None = None
    window: str | None = None
    failure_modes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Lens:
    id: str
    name: str
    question: str
    properties: list[str]


@dataclass(frozen=True)
class Mode:
    id: str
    name: str
    description: str
    min_lens_properties: int | None = None
    requires_lens: str | None = None


@dataclass(frozen=True)
class Registry:
    properties: dict[str, Property]
    lenses: dict[str, Lens]
    modes: dict[str, Mode]

    # --------- loader ---------
    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        path = path or Path(__file__).resolve().parent / "spec.yaml"
        data: dict[str, Any] = yaml.safe_load(path.read_text())

        properties = {
            name: Property(
                name=name,
                level=int(p["level"]),
                description=p.get("description"),
                formula=p.get("formula"),
                source=p.get("source"),
                inputs=list(p.get("inputs", []) or []),
                unit=str(p["unit"]) if p.get("unit") is not None else None,
                granularity=p.get("granularity"),
                window=p.get("window"),
                failure_modes=list(p.get("failure_modes", []) or []),
            )
            for name, p in data["properties"].items()
        }

        lenses = {
            lens["id"]: Lens(
                id=lens["id"],
                name=lens["name"],
                question=lens["question"],
                properties=list(lens["properties"]),
            )
            for lens in data["lenses"]
        }

        modes = {
            m["id"]: Mode(
                id=m["id"],
                name=m["name"],
                description=m.get("description", ""),
                min_lens_properties=m.get("min_lens_properties"),
                requires_lens=m.get("requires_lens"),
            )
            for m in data["ui"]["modes"]
        }

        # integrity check — every lens property must resolve
        for lens in lenses.values():
            for p in lens.properties:
                if p not in properties:
                    raise ValueError(f"Lens {lens.id!r} references unknown property {p!r}")
        return cls(properties=properties, lenses=lenses, modes=modes)

    # --------- helpers ---------
    def lens_properties(self, lens_id: str) -> list[str]:
        return list(self.lenses[lens_id].properties)

    def tooltip(self, name: str) -> str:
        """Markdown-friendly provenance string for a single property.

        Description comes first (italicised) as the human-readable hook;
        technical provenance follows below.
        """
        p = self.properties.get(name)
        if p is None:
            return f"_{name}_ — not in registry."
        lines = []
        if p.description:
            lines.append(f"_{p.description}_")
        lines.append(f"**{p.name}**  · Level {p.level}")
        if p.formula:
            lines.append(f"Formula: `{p.formula}`")
        if p.source:
            lines.append(f"Source: `{p.source}`")
        if p.inputs:
            lines.append(f"Inputs: {', '.join(f'`{i}`' for i in p.inputs)}")
        if p.unit:
            lines.append(f"Unit: {p.unit}")
        if p.window:
            lines.append(f"Window: {p.window}")
        if p.granularity:
            lines.append(f"Granularity: {p.granularity}")
        if p.failure_modes:
            lines.append("Failure modes:")
            lines += [f"  - {fm}" for fm in p.failure_modes]
        return "\n\n".join(lines)

    def short_help(self, name: str) -> str:
        """One-line description (falls back to name if missing)."""
        p = self.properties.get(name)
        if p is None or not p.description:
            return name
        return p.description
