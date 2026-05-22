"""
Pi Script runtime resolver for Melody Maestro — Path B wrapper.

Evaluates session coherence and genre floor policies against a Session entity
built from audio/MIDI analysis results. Mirrors Continuum resolver rule
semantics (equality_rule, range_rule, threshold_rule, context_rule) without
the import dependency. Designed as a drop-in replacement once Continuum ships
as an installable package.

Two IR files, one resolver:
  - runtime_ir.json   — session coherence (BPM range, frequency balance, etc.)
  - genre_rules.json  — genre floor policies (context_rule per genre)

Output shapes two things:
  1. injection_text  — prepended to GPT system prompt before advice generation
  2. resolver_result — rendered as a collapsible audit card below the EP Audit
"""

from __future__ import annotations

import json
import os
from typing import Any

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_IR_PATH  = os.path.join(_ROOT, "runtime_ir.json")
_GENRE_IR_PATH    = os.path.join(_ROOT, "genre_rules.json")

# Cached at module level — files don't change between requests.
_IR:       dict | None = None
_GENRE_IR: dict | None = None


# ── Genre BPM floors and ceilings ─────────────────────────────────────────────
_GENRE_BPM_RANGES: dict[str, tuple[float, float]] = {
    "Rap":                      (120.0, 185.0),
    "Hip-Hop":                  (75.0,  120.0),
    "R&B":                      (55.0,  110.0),
    "Old School R&B / Hip-Hop": (75.0,  105.0),
    "Pop":                      (95.0,  155.0),
    "Alternative Rock":         (95.0,  185.0),
}


# ── IR loaders ────────────────────────────────────────────────────────────────

def _get_ir() -> dict:
    global _IR
    if _IR is None:
        with open(_RUNTIME_IR_PATH, encoding="utf-8") as f:
            _IR = json.load(f)
    return _IR


def _get_genre_ir() -> dict:
    global _GENRE_IR
    if _GENRE_IR is None:
        with open(_GENRE_IR_PATH, encoding="utf-8") as f:
            _GENRE_IR = json.load(f)
    return _GENRE_IR


# ── BPM range helper ──────────────────────────────────────────────────────────

def _bpm_in_genre_range(bpm: float, genre: str) -> bool:
    lo, hi = _GENRE_BPM_RANGES.get(genre, (60.0, 220.0))
    return lo <= bpm <= hi


# ── Entity state builders ─────────────────────────────────────────────────────

def build_audio_session_state(
    bpm: float,
    key: str,
    metrics: dict,
    genre: str,
    session_id: str,
) -> dict:
    return {
        "bpm":                bpm,
        "genre":              genre,
        "key":                key,
        "duration":           metrics.get("duration", 0.0),
        "low_pct":            metrics.get("low_pct", 0.0),
        "mid_pct":            metrics.get("mid_pct", 0.0),
        "high_pct":           metrics.get("high_pct", 0.0),
        "centroid_hz":        metrics.get("centroid_hz", 0.0),
        "bpm_in_genre_range": _bpm_in_genre_range(bpm, genre),
        "upload_type":        "audio",
        "session_id":         session_id,
    }


def build_midi_session_state(
    midi_data: dict,
    genre: str,
    session_id: str,
) -> dict:
    instruments = midi_data.get("instruments", [])
    has_drums = any(i.get("is_drum") for i in instruments)
    melody_note_count = sum(
        i["note_count"] for i in instruments if not i.get("is_drum")
    )
    bpm = float(midi_data.get("bpm", 120.0))
    return {
        "bpm":                bpm,
        "genre":              genre,
        "duration":           float(midi_data.get("duration", 0.0)),
        "bpm_in_genre_range": _bpm_in_genre_range(bpm, genre),
        "has_drums":          has_drums,
        "melody_note_count":  melody_note_count,
        "upload_type":        "midi",
        "session_id":         session_id,
    }


# ── Rule evaluators (matching Continuum resolver semantics exactly) ───────────

def _eval_equality(field: str, value: Any, expected: Any) -> tuple[bool, str]:
    if isinstance(expected, bool):
        actual = value if isinstance(value, bool) else str(value).lower() == "true"
    else:
        actual = value
    if actual != expected:
        return True, f"{field} is {value!r}, expected {expected!r}"
    return False, f"{field} = {value!r}, equals expected {expected!r}"


def _eval_range(
    field: str, value: Any, lo: float | None, hi: float | None
) -> tuple[bool, str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{field} not numeric — skipped"
    if lo is not None and v < lo:
        return True, f"{field} {v} is below minimum {lo}"
    if hi is not None and v > hi:
        return True, f"{field} {v} exceeds maximum {hi}"
    return False, f"{field} = {v}, within range({lo} .. {hi})"


def _eval_threshold(field: str, value: Any, below: float) -> tuple[bool, str]:
    """Violates when value >= below (same as Continuum threshold_rule)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{field} not numeric — skipped"
    if v >= below:
        return True, f"{field} {v} >= threshold {below}"
    return False, f"{field} = {v}, below threshold {below}"


# ── Injection template renderer ───────────────────────────────────────────────

class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def _render_injection(template: str, entity_state: dict) -> str:
    return template.format_map(_SafeDict(entity_state))


# ── Main resolver ─────────────────────────────────────────────────────────────

def run_resolver(entity_state: dict, ir: dict | None = None) -> dict:
    """
    Evaluate all Session constraints against the provided entity state.

    Pass a specific IR dict to use a different policy file (e.g. genre_rules).
    If ir is None, uses runtime_ir.json (session coherence policies).

    Returns:
        {
            "violations":    list[dict],   # constraints that fired as violations
            "passed":        list[dict],   # constraints satisfied
            "active":        list[dict],   # context_rule floors that fired
            "suspended":     list[dict],   # field missing or context not met
            "injection_text": str,         # combined injection for GPT prompt
            "has_violations": bool,
        }
    """
    if ir is None:
        ir = _get_ir()

    constraints_ir = ir.get("constraints", {})
    enforce_list   = ir.get("enforce", {}).get("Session", [])

    violations:      list[dict] = []
    passed:          list[dict] = []
    active:          list[dict] = []
    suspended:       list[dict] = []
    injection_parts: list[str]  = []

    for cname in enforce_list:
        c_ir = constraints_ir.get(cname)
        if not c_ir:
            suspended.append({
                "name":       cname,
                "audit_label": cname,
                "evaluation": "constraint definition not found",
            })
            continue

        rule               = c_ir.get("rule", {})
        rule_kind          = rule.get("kind", "")
        audit_label        = c_ir.get("audit_label", cname)
        injection_template = c_ir.get("injection", "")

        # ── Context filtering ─────────────────────────────────────────────────
        # If a context block is present, all k/v pairs must match entity state.
        # Non-matching constraints are suspended — not violated, not passed.
        context = c_ir.get("context", {})
        if context:
            context_met = all(
                entity_state.get(k) == v for k, v in context.items()
            )
            if not context_met:
                ctx_desc = ", ".join(f"{k}={v!r}" for k, v in context.items())
                suspended.append({
                    "name":       cname,
                    "audit_label": audit_label,
                    "evaluation": f"context not met ({ctx_desc})",
                })
                continue

        # ── context_rule: floor enforcer ──────────────────────────────────────
        # Fires injection whenever context matches. Not a violation — it's a
        # declarative genre floor. Always reports "active", never "violated".
        if rule_kind == "context_rule":
            injection = _render_injection(injection_template, entity_state)
            active.append({
                "name":        cname,
                "audit_label": audit_label,
                "evaluation":  "genre floor active — injection applied",
                "rule_kind":   rule_kind,
            })
            if injection:
                injection_parts.append(injection)
            continue

        # ── Standard rule evaluation ──────────────────────────────────────────
        ref   = rule.get("ref", "")
        field = ref.split(".", 1)[1] if "." in ref else ref

        if field and field not in entity_state:
            suspended.append({
                "name":        cname,
                "audit_label": audit_label,
                "evaluation":  f"field '{field}' not present for this upload type",
            })
            continue

        value = entity_state.get(field)

        if rule_kind == "equality_rule":
            violated, evaluation = _eval_equality(field, value, rule["value"])
        elif rule_kind == "range_rule":
            violated, evaluation = _eval_range(
                field, value, lo=rule.get("lo"), hi=rule.get("hi")
            )
        elif rule_kind == "threshold_rule":
            violated, evaluation = _eval_threshold(field, value, below=rule["below"])
        else:
            suspended.append({
                "name":        cname,
                "audit_label": audit_label,
                "evaluation":  f"unknown rule kind '{rule_kind}'",
            })
            continue

        result = {
            "name":        cname,
            "audit_label": audit_label,
            "evaluation":  evaluation,
            "rule_kind":   rule_kind,
        }

        if violated:
            injection = _render_injection(injection_template, entity_state)
            result["injection"] = injection
            violations.append(result)
            if injection:
                injection_parts.append(injection)
        else:
            passed.append(result)

    return {
        "violations":     violations,
        "passed":         passed,
        "active":         active,
        "suspended":      suspended,
        "injection_text": "\n".join(injection_parts),
        "has_violations": len(violations) > 0,
    }


def run_genre_resolver(entity_state: dict) -> dict:
    """Evaluate genre floor policies from genre_rules.json."""
    return run_resolver(entity_state, ir=_get_genre_ir())
