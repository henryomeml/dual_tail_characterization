"""Validated GUI/controller schemas. No widgets and no silent generator fallback."""
from dataclasses import dataclass
import copy
import json
import math
from .config import Config
from .lane_benchmark import original_lane_records
from .shared_endpoints import effective_lanes, kappa_for_duration

MODES = ("independent_lanes", "shared_endpoints", "history_endpoints", "continuous_filter")
MODE_NAMES = dict(independent_lanes="Independent-lane benchmark", shared_endpoints="Option 1 — Shared symbol endpoints",
                  history_endpoints="Option 2 — History-dependent endpoints", continuous_filter="Continuous filter")


def number(value, field, integer=False):
    try:
        if isinstance(value, bool):
            raise ValueError()
        parsed = float(value)
        if not math.isfinite(parsed) or integer and not parsed.is_integer():
            raise ValueError()
        return int(parsed) if integer else parsed
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field}: expected a finite {'integer' if integer else 'number'}") from exc


def complete_pairs(rows, n, name, fields):
    result, seen = [], set()
    for i, row in enumerate(rows, 1):
        if set(row) != set(fields):
            raise ValueError(f"{name} row {i}: columns must be {', '.join(fields)}")
        a, b = (number(row[k], f"{name} row {i} {k}", True) for k in ("start", "end"))
        if not (0 <= a < n and 0 <= b < n) or (a,b) in seen:
            raise ValueError(f"{name} row {i}: invalid or duplicate ordered pair {a}->{b}")
        seen.add((a,b))
        result.append(dict(start=a, end=b, **{k:number(row[k], f"{name} row {i} {k}") for k in fields if k not in ("start", "end")}))
    if seen != {(a,b) for a in range(n) for b in range(n)}:
        raise ValueError(f"{name}: exactly one row per ordered pair is required ({n*n} rows)")
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    snapshot_json: str

    @classmethod
    def parse(cls, raw):
        raw = copy.deepcopy(raw)
        expected = {"mode", "base", "levels", "lanes", "incoming", "outgoing", "contexts", "noise_model", "preset_name", "output_root"}
        if set(raw) != expected:
            raise ValueError(f"configuration fields missing/unknown: {sorted(set(raw)^expected)}")
        mode = raw["mode"]
        if mode not in MODES:
            raise ValueError(f"mode: unsupported generator {mode!r}")
        if not isinstance(raw["output_root"],str) or not raw["output_root"].strip():
            raise ValueError("Output root directory must be nonempty")
        levels = raw["levels"]
        if len(levels) < 2:
            raise ValueError("Level Properties: at least two rows required")
        for i, row in enumerate(levels):
            if set(row) != {"level_id", "mean", "sigma", "probability"}:
                raise ValueError(f"Level Properties row {i+1}: invalid columns")
            if number(row["level_id"], f"Level Properties row {i+1} level_id", True) != i:
                raise ValueError(f"Level Properties row {i+1}: level_id must be {i}")
            for key in ("mean", "sigma", "probability"):
                row[key] = number(row[key], f"Level Properties row {i+1} {key}")
            row["level_id"] = i
            if row["sigma"] < 0 or row["probability"] < 0:
                raise ValueError(f"Level Properties row {i+1}: sigma and probability must be nonnegative")
        base = raw["base"]
        if mode in ("shared_endpoints", "history_endpoints"):
            if base.get("channel_bandwidth_hz") is not None or base.get("noise_std", 0) != 0:
                raise ValueError(f"{mode}: channel and additive sample noise are unsupported; disable them explicitly")
        if mode == "independent_lanes" and (base.get("channel_bandwidth_hz") is not None or base.get("symbol_mode") != "random"):
            raise ValueError("independent_lanes: no channel or transmitted explicit sequence; use lane occurrence weights")
        # The existing shared connector clock is reused explicitly by Option 2.
        base["waveform_mode"] = "shared_endpoints" if mode in ("shared_endpoints", "history_endpoints") else "continuous_filter"
        base["levels"] = [row["mean"] for row in levels]
        base["endpoint_sigmas"] = [row["sigma"] for row in levels] if base["waveform_mode"] == "shared_endpoints" else None
        if base["waveform_mode"] == "continuous_filter":
            base["shared_kappa"] = None
        base["probabilities"] = [row["probability"] for row in levels] if base.get("symbol_mode", "random") == "random" else None
        try:
            config = Config(**base)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"General / impairment settings: {exc}") from exc
        if mode != "independent_lanes" and config.n_symbols < 3:
            raise ValueError("Number of symbols: at least 3 required for a complete 2UI GUI view")
        if abs(sum(row["probability"] for row in levels)-1) > 1e-12:
            raise ValueError("Level Properties probability column must sum to one within 1e-12")
        raw["base"] = config.to_dict()
        n = len(levels)
        if mode == "independent_lanes":
            fields = ("start", "end", "mean_start", "mean_end", "sigma", "kappa", "center", "weight")
            raw["lanes"] = complete_pairs(raw["lanes"], n, "Lane Properties", fields)
            for i, row in enumerate(raw["lanes"], 1):
                if row["sigma"] < 0 or row["kappa"] <= 0 or not 0 <= row["center"] <= 1 or row["weight"] < 0:
                    raise ValueError(f"Lane Properties row {i}: sigma/weight >= 0, kappa > 0 and center in [0,1] required")
            if abs(sum(r["weight"] for r in raw["lanes"])-1) > 1e-12:
                raise ValueError("Lane Properties occurrence weights must sum to one within 1e-12")
            if raw["noise_model"] not in ("trace_offset", "sample_independent"):
                raise ValueError("noise_model must be trace_offset or sample_independent")
        if mode == "history_endpoints":
            for kind in ("incoming", "outgoing"):
                raw[kind] = complete_pairs(raw[kind], n, f"{kind.title()} proposals", ("start", "end", "mean"))
            contexts, seen = [], set()
            for i, row in enumerate(raw["contexts"], 1):
                if set(row) != {"previous", "current", "next", "mean", "sigma"}:
                    raise ValueError(f"Context override row {i}: invalid columns")
                key = tuple(number(row[k], f"Context override row {i} {k}", True) for k in ("previous", "current", "next"))
                if any(v < 0 or v >= n for v in key) or key in seen:
                    raise ValueError(f"Context override row {i}: invalid or duplicate context {key}")
                seen.add(key)
                sigma = None if row["sigma"] in (None, "") else number(row["sigma"], f"Context override row {i} sigma")
                if sigma is not None and sigma < 0:
                    raise ValueError(f"Context override row {i}: sigma must be nonnegative")
                contexts.append(dict(zip(("previous", "current", "next"), key), mean=number(row["mean"], f"Context override row {i} mean"), sigma=sigma))
            raw["contexts"] = contexts
        return cls(json.dumps(raw, sort_keys=True, allow_nan=False))

    def to_dict(self):
        return json.loads(self.snapshot_json)

    @property
    def mode(self):
        return self.to_dict()["mode"]

    @property
    def config(self):
        return Config(**self.to_dict()["base"])


def preset(mode="shared_endpoints", n=4):
    if mode not in MODES or not isinstance(n, int) or n < 2:
        raise ValueError("invalid preset mode or level count")
    original = original_lane_records() if n == 4 else [dict(start=a,end=b,mean_start=float(a),mean_end=float(b),sigma=.1) for a in range(n) for b in range(n)]
    levels = [dict(level_id=i,mean=float(i),sigma=.2 if i in (0,n-1) else .1,probability=1/n) for i in range(n)]
    base = Config(n_symbols=2000).to_dict()
    return ExperimentConfig.parse(dict(mode=mode,base=base,levels=levels,
        lanes=[dict(r,kappa=22.,center=.5,weight=1/(n*n)) for r in original],
        incoming=[dict(start=r["start"],end=r["end"],mean=r["mean_end"]) for r in original],
        outgoing=[dict(start=r["start"],end=r["end"],mean=r["mean_start"]) for r in original],
        contexts=[],noise_model="trace_offset",preset_name=f"{MODE_NAMES[mode]} PAM{n}",output_root="runs/gui"))


def capabilities(mode):
    if mode not in MODES:
        raise ValueError("unsupported mode")
    return dict(lane_edit=mode=="independent_lanes", proposals_edit=mode=="history_endpoints",
                contexts_edit=mode=="history_endpoints", endpoint_sigma_edit=mode in ("shared_endpoints","history_endpoints"),
                probability_edit=mode!="independent_lanes", channel_edit=mode=="continuous_filter",
                additive_edit=mode in ("continuous_filter","independent_lanes"), explicit_edit=mode!="independent_lanes",
                per_lane_timing_edit=mode=="independent_lanes")


def lane_preview(snapshot):
    raw, c = snapshot.to_dict(), snapshot.config
    if snapshot.mode == "independent_lanes":
        return raw["lanes"]
    kappa = c.shared_kappa or (kappa_for_duration(c.transition_10_90_ui) if snapshot.mode != "continuous_filter" else 2*math.log(9)/c.transition_10_90_ui)
    if snapshot.mode == "history_endpoints":
        return [dict(start=a,end=b,mean_start="context dependent",mean_end="context dependent",sigma_start="level/context",sigma_end="level/context",kappa=kappa,center=.5,weight="actual contexts after generation") for a in range(len(c.levels)) for b in range(len(c.levels))]
    if snapshot.mode == "continuous_filter":
        return [dict(start=a,end=b,mean_start=c.levels[a],mean_end=c.levels[b],sigma_start="not endpoint noise",sigma_end="not endpoint noise",kappa="smoothing kernel",center=.5,weight="source sequence") for a in range(len(c.levels)) for b in range(len(c.levels))]
    return [dict(r,kappa=kappa,center=.5,weight="source sequence") for r in effective_lanes(c.levels,c.endpoint_sigmas)]
