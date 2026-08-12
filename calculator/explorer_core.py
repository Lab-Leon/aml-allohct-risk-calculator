from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "calculator" / "input_schema.json"
METADATA_PATH = ROOT / "outputs" / "model_metadata.json"
THRESHOLDS_PATH = ROOT / "calculator" / "risk_thresholds.json"
COMPACT_MANIFEST_PATH = ROOT / "calculator" / "compact_rsf" / "manifest.json"

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
THRESHOLDS = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))

FEATURES = list(SCHEMA["features"])
CATEGORICAL = [feature for feature in FEATURES if feature != "age"]
OUTCOMES = ["OS", "Relapse", "DFS", "TRM_NRM"]
MODELS = ["CoxPH", "Elastic-net Cox", "RSF"]
METADATA_BY_KEY = {(row["outcome"], row["model"]): row for row in METADATA}


class CompactRSF:
    def __init__(self, path: Path | io.BytesIO):
        with np.load(path, allow_pickle=False) as archive:
            self.payload = {key: archive[key] for key in archive.files}
        self.outcome = str(self.payload["outcome"][0])
        self.categories = {
            feature: json.loads(str(self.payload[f"categories_{feature}"][0]))
            for feature in CATEGORICAL
        }

    def _encode(self, row: pd.DataFrame) -> np.ndarray:
        values = row.iloc[0]
        age = (float(values["age"]) - float(self.payload["age_mean"][0])) / float(
            self.payload["age_scale"][0]
        )
        encoded = [age]
        for feature in CATEGORICAL:
            encoded.extend(
                float(values[feature] == level) for level in self.categories[feature]
            )
        return np.asarray(encoded, dtype=np.float32)

    def predict_outputs(self, row: pd.DataFrame) -> tuple[float, list[float]]:
        encoded = self._encode(row)
        payload = self.payload
        offsets = payload["tree_offsets"]
        left = payload["children_left"]
        right = payload["children_right"]
        feature = payload["feature"]
        threshold = payload["threshold"]
        risk = payload["leaf_risk"]
        survival = payload["leaf_survival"]
        risk_total = 0.0
        survival_total = np.zeros(survival.shape[1], dtype=np.float64)
        for tree_index in range(len(offsets) - 1):
            node = int(offsets[tree_index])
            while left[node] != -1:
                node = int(
                    left[node]
                    if encoded[int(feature[node])] <= threshold[node]
                    else right[node]
                )
            risk_total += float(risk[node])
            if survival.shape[1]:
                survival_total += survival[node]
        trees = len(offsets) - 1
        return risk_total / trees, (survival_total / trees).tolist()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def repository_path(relative_path: str) -> Path:
    """Resolve registry paths consistently on Windows and Linux."""
    return ROOT.joinpath(*relative_path.replace("\\", "/").split("/"))


def compact_artifact(entry: dict) -> Path | io.BytesIO:
    if "compact_parts" not in entry:
        return repository_path(entry["compact_file"])
    payload = b"".join(
        repository_path(relative_path).read_bytes()
        for relative_path in entry["compact_parts"]
    )
    return io.BytesIO(payload)


def compact_artifact_sha256(entry: dict) -> str | None:
    if "compact_parts" not in entry:
        path = repository_path(entry["compact_file"])
        return sha256(path) if path.exists() else None
    digest = hashlib.sha256()
    for relative_path in entry["compact_parts"]:
        path = repository_path(relative_path)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest().lower()


def expected_categories() -> dict[str, list[str]]:
    return {
        feature: list(SCHEMA["categorical"][feature]["model_levels"])
        for feature in CATEGORICAL
    }


def threshold_key(outcome: str, model_name: str) -> str:
    return f"{outcome}|{model_name}"


def build_default_values() -> dict[str, object]:
    values: dict[str, object] = {"age": SCHEMA["numeric"]["age"]["default"]}
    values.update(
        {
            feature: SCHEMA["categorical"][feature]["options"][0]
            for feature in CATEGORICAL
        }
    )
    return values


def load_model_file(outcome: str, model_name: str, *, validate_encoder: bool = True):
    entry = METADATA_BY_KEY[(outcome, model_name)]
    if model_name == "RSF" and COMPACT_MANIFEST_PATH.exists():
        manifest = json.loads(COMPACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        compact = next(item for item in manifest["entries"] if item["outcome"] == outcome)
        model = CompactRSF(compact_artifact(compact))
        if validate_encoder and model.categories != expected_categories():
            raise RuntimeError(f"encoder level mismatch: {outcome}/{model_name}")
        return model
    model = joblib.load(repository_path(entry["model_file"]))
    if validate_encoder:
        fitted = {
            feature: [str(value) for value in values]
            for feature, values in zip(
                CATEGORICAL,
                model.named_steps["prep"].named_transformers_["cat"].categories_,
            )
        }
        if fitted != expected_categories():
            raise RuntimeError(f"encoder level mismatch: {outcome}/{model_name}")
    return model


def _validate_schema_options(failures: list[str]) -> None:
    for feature in CATEGORICAL:
        specification = SCHEMA["categorical"][feature]
        options = list(specification["options"])
        model_levels = list(specification["model_levels"])
        unsupported = [value for value in options if value not in model_levels]
        zero_support = [
            value
            for value in options
            if int(specification["development_counts"].get(value, 0)) <= 0
        ]
        if unsupported:
            failures.append(f"UI option absent from model levels: {feature}={unsupported}")
        if zero_support:
            failures.append(f"UI option has zero development support: {feature}={zero_support}")


def _validate_thresholds(
    outcome: str, model_name: str, entry: dict, failures: list[str]
) -> None:
    key = threshold_key(outcome, model_name)
    if key not in THRESHOLDS:
        failures.append(f"missing thresholds: {key}")
        return
    info = THRESHOLDS[key]
    if info.get("model_sha256") != entry.get("model_sha256"):
        failures.append(f"threshold/model hash mismatch: {key}")
    scores = np.asarray(info.get("development_scores_sorted", []), dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        failures.append(f"invalid development-score registry: {key}")
    elif np.any(np.diff(scores) < 0):
        failures.append(f"unsorted development-score registry: {key}")
    low = float(info.get("low_intermediate", np.nan))
    high = float(info.get("intermediate_high", np.nan))
    if not (np.isfinite(low) and np.isfinite(high) and low <= high):
        failures.append(f"invalid risk cutoffs: {key}")


def validate_model_schema_contract() -> dict[str, object]:
    actual_schema_hash = sha256(SCHEMA_PATH)
    expected = expected_categories()
    expected_keys = {(outcome, model) for outcome in OUTCOMES for model in MODELS}
    failures: list[str] = []
    _validate_schema_options(failures)

    missing = sorted(expected_keys - set(METADATA_BY_KEY))
    extra = sorted(set(METADATA_BY_KEY) - expected_keys)
    if missing:
        failures.append(f"missing model metadata: {missing}")
    if extra:
        failures.append(f"unexpected model metadata: {extra}")

    validated = 0
    compact_entries: dict[str, dict] = {}
    if COMPACT_MANIFEST_PATH.exists():
        compact_manifest = json.loads(COMPACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        compact_entries = {entry["outcome"]: entry for entry in compact_manifest["entries"]}
    for outcome, model_name in sorted(expected_keys):
        entry = METADATA_BY_KEY.get((outcome, model_name))
        if entry is None:
            continue
        if entry.get("calculator_schema_sha256") != actual_schema_hash:
            failures.append(f"schema hash mismatch: {outcome}/{model_name}")
            continue
        if entry.get("features") != FEATURES:
            failures.append(f"feature-order mismatch: {outcome}/{model_name}")
            continue
        if entry.get("categorical_levels") != expected:
            failures.append(f"metadata level mismatch: {outcome}/{model_name}")
            continue
        _validate_thresholds(outcome, model_name, entry, failures)
        if model_name == "RSF":
            compact = compact_entries.get(outcome)
            if compact is None:
                model_path = repository_path(entry["model_file"])
                if not model_path.exists() or sha256(model_path) != entry.get("model_sha256"):
                    failures.append(f"model hash mismatch: {outcome}/{model_name}")
            else:
                if compact.get("source_model_sha256") != entry.get("model_sha256"):
                    failures.append(f"compact/source hash mismatch: {outcome}")
                elif compact.get("calculator_schema_sha256") != actual_schema_hash:
                    failures.append(f"compact/schema hash mismatch: {outcome}")
                elif compact_artifact_sha256(compact) != compact.get("compact_sha256"):
                    failures.append(f"compact artifact hash mismatch: {outcome}")
        else:
            model_path = repository_path(entry["model_file"])
            if not model_path.exists() or sha256(model_path) != entry.get("model_sha256"):
                failures.append(f"model hash mismatch: {outcome}/{model_name}")
        validated += 1

    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "schema_sha256": actual_schema_hash,
        "models_validated": validated,
    }


def model_score(model, row: pd.DataFrame) -> float:
    if isinstance(model, CompactRSF):
        value = float(model.predict_outputs(row)[0])
    else:
        value = float(np.asarray(model.predict(row), dtype=float).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("Model returned a non-finite risk score")
    return value


def survival_estimates(
    model, row: pd.DataFrame, times: tuple[float, ...] = (12.0, 24.0, 36.0)
) -> list[float]:
    if isinstance(model, CompactRSF):
        compact_times = tuple(float(value) for value in model.payload["horizons"])
        if tuple(times) != compact_times:
            raise ValueError(f"Compact RSF supports horizons {compact_times}, not {times}")
        values = [float(value) for value in model.predict_outputs(row)[1]]
    else:
        encoded = model.named_steps["prep"].transform(row)
        function = model.named_steps["model"].predict_survival_function(encoded)[0]
        values = [float(function(time)) for time in times]
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Model returned an invalid survival probability")
    return values


def risk_rank(
    score_value: float, outcome: str, model_name: str
) -> tuple[float, str]:
    info = THRESHOLDS[threshold_key(outcome, model_name)]
    development = np.asarray(info["development_scores_sorted"], dtype=float)
    percentile = float(
        np.searchsorted(development, score_value, side="right")
        / development.size
        * 100.0
    )
    if score_value <= float(info["low_intermediate"]):
        group = "Low"
    elif score_value <= float(info["intermediate_high"]):
        group = "Intermediate"
    else:
        group = "High"
    return percentile, group
