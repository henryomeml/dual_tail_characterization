"""Controlled history-dependent endpoints, with explicit non-wrapping guards."""
from dataclasses import dataclass
import numpy as np
from .experiment_config import ExperimentConfig
from .shared_endpoints import assemble_shared_waveform


@dataclass(frozen=True)
class HistoryResult:
    waveform: object
    contexts: np.ndarray
    endpoint_means: np.ndarray
    endpoint_sigmas: np.ndarray
    guards: np.ndarray


def context_parameters(snapshot, previous, current, following):
    raw = snapshot.to_dict()
    incoming = {(r["start"],r["end"]):r["mean"] for r in raw["incoming"]}
    outgoing = {(r["start"],r["end"]):r["mean"] for r in raw["outgoing"]}
    overrides = {(r["previous"],r["current"],r["next"]):r for r in raw["contexts"]}
    means, sigmas = [], []
    for a,b,c in zip(previous,current,following):
        row = overrides.get((int(a),int(b),int(c)))
        means.append(row["mean"] if row else (incoming[a,b]+outgoing[b,c])/2)
        sigmas.append(row["sigma"] if row and row["sigma"] is not None else raw["levels"][b]["sigma"])
    return np.asarray(means), np.asarray(sigmas)


def generate_history_endpoints(snapshot: ExperimentConfig, pattern_result=None):
    if snapshot.mode != "history_endpoints":
        raise ValueError("history generator requires history_endpoints configuration")
    c = snapshot.config
    symbol_rng, endpoint_rng, _noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3)]
    if pattern_result is not None:
        from dataclasses import replace
        symbols=pattern_result.symbols.copy()
        guards=np.array([pattern_result.before[-1],pattern_result.after[0]])
        c=replace(c,symbol_mode="explicit",sequence=tuple(map(int,symbols)),probabilities=None)
    elif c.symbol_mode == "random":
        symbols = symbol_rng.choice(len(c.levels), c.n_symbols, p=c.probabilities)
        guards = symbol_rng.choice(len(c.levels), 2, p=c.probabilities)
    else:
        symbols = np.asarray(c.sequence, dtype=int)
        guards = symbols[[0,-1]].copy()
    extended = np.concatenate((guards[:1],symbols,guards[1:]))
    contexts = np.stack((extended[:-2],extended[1:-1],extended[2:]),axis=1)
    means, sigmas = context_parameters(snapshot,*contexts.T)
    z = endpoint_rng.standard_normal(c.n_symbols)
    endpoints = means + sigmas*z
    metadata = dict(generator="history_endpoints", context_rule="M(a,b,c) = (incoming(a,b)+outgoing(b,c))/2; explicit overrides take precedence",
        history_model="controlled history dependence; does not reproduce arbitrary independent-lane endpoints exactly",
        context_guards=guards.tolist(), context_guard_policy="independent draws after retained symbols" if c.symbol_mode=="random" else "constant endpoint extension; no wraparound",
        effective_lane_reference="actual observed adjacent context-pair weights; not equal-context averaging")
    if pattern_result is not None:
        metadata.update(pattern=pattern_result.metadata,context_guard_policy=pattern_result.metadata["guard_policy"])
    w = assemble_shared_waveform(c,symbols,endpoints,z,metadata,lane_parameters=snapshot.to_dict().get("lane_overrides"))
    return HistoryResult(w,contexts,means,sigmas,guards)
