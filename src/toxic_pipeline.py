from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:
    import contractions
except ImportError:  # pragma: no cover - optional dependency
    contractions = None

LABEL_COLUMNS_TO_DROP = [
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

FEATURE_COLUMNS = [
    "Question Mark Count",
    "Profanity Count",
    "Repeated Punctuation Count",
    "Short/Unclear Without Toxic Signal Flag",
    "Second-person Pronoun Count",
    "URL Count",
    "Non-toxic Negation Pattern Count",
]

ALL_ENGINEERED_FEATURE_COLUMNS = [
    "Character Count",
    "Word Count",
    "Exclamation Count",
    "Profanity Count",
    "Strong Toxic Signal Flag",
    "Second-person Pronoun Count",
    "Repeated Character Pattern Count",
    "Average Word Length",
    "Uppercase Ratio",
    "Question Mark Count",
    "Repeated Punctuation Count",
    "Identity-group Term Count",
    "URL Count",
    "Negation Count",
    "Non-toxic Negation Pattern Count",
    "Short/Unclear Without Toxic Signal Flag",
]

WORD_TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "stop_words": list(
        set(ENGLISH_STOP_WORDS)
        - {"not", "no", "never"}
    ),
}

BEST_LOGISTIC_REGRESSION_CONFIG = {
    "penalty": "l2",
    "solver": "liblinear",
    "C": 0.6719567619175353,
    "class_weight": {0: 1, 1: 3},
    "tol": 1.096675267822891e-05,
    "max_iter": 2000,
    "random_state": 42,
    "n_jobs": None,
}

MODEL_FILENAME = "best_model_final.pkl"
WORD_VECTORIZER_FILENAME = "best_model_word_vectorizer.pkl"
SCALER_FILENAME = "best_model_scaler.pkl"
METADATA_FILENAME = "best_model_metadata.pkl"

FALLBACK_MODEL_FILENAME = "optuna_feature_test_best_model.pkl"
FALLBACK_WORD_VECTORIZER_FILENAME = "optuna_feature_test_word_vectorizer.pkl"
FALLBACK_SCALER_FILENAME = "optuna_feature_test_scaler.pkl"
FALLBACK_METADATA_FILENAME = "optuna_feature_test_metadata.pkl"

PROFANITY_TERMS = [
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "bastard",
    "asshole",
    "idiot",
    "moron",
    "dumb",
    "stupid",
    "suck",
    "crap",
    "damn",
    "jerk",
    "loser",
    "trash",
]

IDENTITY_TERMS = [
    "black",
    "white",
    "gay",
    "lesbian",
    "transgender",
    "trans",
    "muslim",
    "jewish",
    "christian",
    "hispanic",
    "asian",
    "woman",
    "women",
    "man",
    "men",
]

# Temporarily exclude "you" from the direct second-person signal while keeping
# the existing feature schema unchanged.
SECOND_PERSON_TERMS = ["your", "yours", "yourself"]
NEGATION_TERMS = ["not", "never", "no", "none", "cannot", "cant", "do not"]
NON_TOXIC_NEGATION_PATTERNS = [
    r"\bnot\s+(?:stupid|dumb|idiot|moron|trash|wrong|bad|terrible|awful|useless)\b",
    r"\bnot\s+(?:an|a)\s+(?:idiot|moron|loser|bastard|fool)\b",
    r"\bdo\s+not\s+(?:like|love|agree|hate|dislike|attack|insult|blame)\b",
    r"\bcannot\s+(?:hate|blame)\b",
    r"\bnot\s+trying\s+to\s+(?:attack|insult|offend)\b",
]
AFFECTIONATE_PROFANITY_PATTERNS = [
    r"\bi\s+fucking\s+love\s+(?:you|this|it|that|them|him|her)\b",
    r"\bi\s+fucking\s+(?:love|adore|like)\b",
    r"\bfucking\s+love\s+(?:you|this|it|that|them|him|her)\b",
    r"\b(?:this|that|it)\s+is\s+fucking\s+(?:awesome|amazing|great|wonderful|fantastic|beautiful|lovely|cool|perfect|excellent|incredible)\b",
    r"\bfucking\s+(?:awesome|amazing|great|wonderful|fantastic|beautiful|lovely|cool|perfect|excellent|incredible)\b",
]
NON_TOXIC_NEGATION_PROTECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in NON_TOXIC_NEGATION_PATTERNS
]
AFFECTIONATE_PROFANITY_PROTECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in AFFECTIONATE_PROFANITY_PATTERNS
]
NON_TOXIC_NEGATION_PLACEHOLDER = " supportivephrase "
AFFECTIONATE_PROFANITY_PLACEHOLDER = " positiveemphasis "
COMMON_SHORT_TOKENS = {
    "i",
    "me",
    "my",
    "your",
    "yours",
    "yourself",
    "it",
    "this",
    "that",
    "a",
    "an",
    "the",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "and",
}

SHORT_NEUTRAL_TOKENS = COMMON_SHORT_TOKENS | {
    "hi",
    "hello",
    "thanks",
    "thank",
    "please",
    "ok",
    "okay",
    "yes",
    "nope",
    "yep",
    "u",
}


def resolve_src_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    return resolve_src_dir().parent


def resolve_data_path() -> Path:
    return resolve_project_root() / "data" / "train.csv"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_contractions(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if contractions is not None:
        try:
            return contractions.fix(text)
        except IndexError:
            normalized_text = text.replace("\u2018", "'").replace("\u2019", "'")
            if normalized_text != text:
                try:
                    return contractions.fix(normalized_text)
                except IndexError:
                    pass
            # Some multilingual or malformed strings can trigger an internal
            # bounds-check bug in `contractions`/`textsearch`. Falling back to
            # the original text keeps feature generation resilient.
            return text
    return text


def protect_non_toxic_negations(text: str) -> str:
    if not isinstance(text, str):
        return ""

    protected_text = expand_contractions(text)
    for pattern in AFFECTIONATE_PROFANITY_PROTECTION_PATTERNS:
        protected_text = pattern.sub(AFFECTIONATE_PROFANITY_PLACEHOLDER, protected_text)
    for pattern in NON_TOXIC_NEGATION_PROTECTION_PATTERNS:
        protected_text = pattern.sub(NON_TOXIC_NEGATION_PLACEHOLDER, protected_text)
    return re.sub(r"\s+", " ", protected_text).strip()


def make_term_pattern(terms: list[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(term) for term in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


PROFANITY_PATTERN = make_term_pattern(PROFANITY_TERMS)
IDENTITY_PATTERN = make_term_pattern(IDENTITY_TERMS)
SECOND_PERSON_PATTERN = make_term_pattern(SECOND_PERSON_TERMS)
NEGATION_PATTERN = make_term_pattern(NEGATION_TERMS)
NON_TOXIC_NEGATION_PATTERN = re.compile("|".join(NON_TOXIC_NEGATION_PATTERNS))


def count_pattern(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(str(text).lower()))


def repeated_characters_score(text: str) -> int:
    return len(re.findall(r"(.)\1{2,}", str(text).lower()))


def uppercase_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", str(text))
    if not letters:
        return 0.0
    uppercase_letters = sum(1 for char in letters if char.isupper())
    return uppercase_letters / len(letters)


def repeated_punctuation_count(text: str) -> int:
    return len(re.findall(r"([!?.,])\1+", str(text)))


def short_unclear_without_toxic_signal(cleaned_text: str, profanity_count: int) -> int:
    tokens = str(cleaned_text).split()
    content_tokens = [token for token in tokens if token not in COMMON_SHORT_TOKENS]
    too_short = len(tokens) < 3 or len(content_tokens) < 1
    return int(too_short and profanity_count == 0)


def should_override_short_neutral_input(raw_text: str) -> bool:
    clean = clean_text(raw_text)
    tokens = clean.split()
    if not tokens or len(tokens) > 2:
        return False

    profanity_count = count_pattern(clean, PROFANITY_PATTERN)
    if profanity_count > 0:
        return False

    return all(token in SHORT_NEUTRAL_TOKENS for token in tokens)


def build_all_engineered_features(raw_text: str) -> pd.DataFrame:
    raw_text = str(raw_text)
    protected_text = protect_non_toxic_negations(raw_text)
    clean = clean_text(protected_text)
    clean_original = clean_text(raw_text)
    profanity = count_pattern(clean, PROFANITY_PATTERN)
    words = clean.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0.0

    row = {
        "Character Count": len(raw_text),
        "Word Count": len(words),
        "Exclamation Count": raw_text.count("!"),
        "Profanity Count": profanity,
        "Strong Toxic Signal Flag": int(profanity > 0),
        "Second-person Pronoun Count": count_pattern(clean, SECOND_PERSON_PATTERN),
        "Repeated Character Pattern Count": repeated_characters_score(raw_text),
        "Average Word Length": avg_word_length,
        "Uppercase Ratio": uppercase_ratio(raw_text),
        "Question Mark Count": raw_text.count("?"),
        "Repeated Punctuation Count": repeated_punctuation_count(raw_text),
        "Identity-group Term Count": count_pattern(clean, IDENTITY_PATTERN),
        "URL Count": len(re.findall(r"http\S+|www\S+", raw_text)),
        "Negation Count": count_pattern(clean_original, NEGATION_PATTERN),
        "Non-toxic Negation Pattern Count": count_pattern(clean_original, NON_TOXIC_NEGATION_PATTERN),
        "Short/Unclear Without Toxic Signal Flag": short_unclear_without_toxic_signal(clean, profanity),
    }
    return pd.DataFrame([row], columns=ALL_ENGINEERED_FEATURE_COLUMNS)


def build_engineered_features(raw_text: str) -> pd.DataFrame:
    return build_all_engineered_features(raw_text)[FEATURE_COLUMNS]


def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(**WORD_TFIDF_CONFIG)


def build_scaler() -> StandardScaler:
    return StandardScaler(with_mean=False)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _save_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file)


def _reduce_scaler_to_features(
    scaler: StandardScaler,
    source_columns: list[str],
    target_columns: list[str],
) -> StandardScaler:
    index_lookup = [source_columns.index(column) for column in target_columns]
    reduced_scaler = StandardScaler(with_mean=False)
    reduced_scaler.n_features_in_ = len(target_columns)

    if hasattr(scaler, "scale_"):
        reduced_scaler.scale_ = scaler.scale_[index_lookup]
    if hasattr(scaler, "var_"):
        reduced_scaler.var_ = scaler.var_[index_lookup]
    if hasattr(scaler, "mean_"):
        reduced_scaler.mean_ = scaler.mean_[index_lookup]
    if hasattr(scaler, "n_samples_seen_"):
        reduced_scaler.n_samples_seen_ = scaler.n_samples_seen_

    return reduced_scaler


def _resolve_existing_path(preferred: str, fallback: str | None = None) -> Path:
    src_dir = resolve_src_dir()
    preferred_path = src_dir / preferred
    if preferred_path.exists():
        return preferred_path
    if fallback is not None:
        fallback_path = src_dir / fallback
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(
        f"Could not find '{preferred}'"
        + (f" or fallback '{fallback}'." if fallback is not None else ".")
    )


def save_best_artifacts(
    model: Any,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer | StandardScaler | None = None,
    scaler: StandardScaler | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if scaler is None and isinstance(char_vectorizer, StandardScaler):
        scaler = char_vectorizer
        char_vectorizer = None

    if scaler is None:
        raise ValueError("A fitted scaler is required when saving artifacts.")

    src_dir = resolve_src_dir()
    metadata_to_save = {
        "feature_columns": FEATURE_COLUMNS,
        "word_tfidf_config": WORD_TFIDF_CONFIG,
        "uses_char_vectorizer": False,
        "text_feature_blocks": ["word_tfidf"],
        "best_logistic_regression_config": BEST_LOGISTIC_REGRESSION_CONFIG,
    }
    if metadata:
        metadata_to_save.update(metadata)

    paths = {
        "model": src_dir / MODEL_FILENAME,
        "word_vectorizer": src_dir / WORD_VECTORIZER_FILENAME,
        "scaler": src_dir / SCALER_FILENAME,
        "metadata": src_dir / METADATA_FILENAME,
    }

    _save_pickle(paths["model"], model)
    _save_pickle(paths["word_vectorizer"], word_vectorizer)
    _save_pickle(paths["scaler"], scaler)
    _save_pickle(paths["metadata"], metadata_to_save)

    return paths


def load_best_artifacts() -> dict[str, Any]:
    model_path = _resolve_existing_path(MODEL_FILENAME, FALLBACK_MODEL_FILENAME)
    word_vectorizer_path = _resolve_existing_path(
        WORD_VECTORIZER_FILENAME,
        FALLBACK_WORD_VECTORIZER_FILENAME,
    )
    scaler_path = _resolve_existing_path(SCALER_FILENAME, FALLBACK_SCALER_FILENAME)

    metadata_path: Path | None = None
    try:
        metadata_path = _resolve_existing_path(METADATA_FILENAME, FALLBACK_METADATA_FILENAME)
    except FileNotFoundError:
        metadata_path = None

    scaler = _load_pickle(scaler_path)
    metadata = _load_pickle(metadata_path) if metadata_path is not None else {}

    selected_features = metadata.get("feature_columns") or metadata.get("best_selected_features") or FEATURE_COLUMNS

    if getattr(scaler, "n_features_in_", len(selected_features)) == len(ALL_ENGINEERED_FEATURE_COLUMNS):
        scaler = _reduce_scaler_to_features(
            scaler=scaler,
            source_columns=ALL_ENGINEERED_FEATURE_COLUMNS,
            target_columns=selected_features,
        )

    artifacts = {
        "model": _load_pickle(model_path),
        "word_vectorizer": _load_pickle(word_vectorizer_path),
        "scaler": scaler,
        "feature_columns": selected_features,
        "metadata": metadata,
        "paths": {
            "model": model_path,
            "word_vectorizer": word_vectorizer_path,
            "scaler": scaler_path,
            "metadata": metadata_path,
        },
    }
    artifacts = _normalize_prediction_artifacts(artifacts)
    _validate_artifact_feature_contract(artifacts)
    return artifacts


def _infer_feature_columns(artifacts: dict[str, Any]) -> list[str]:
    explicit_columns = artifacts.get("feature_columns")
    if explicit_columns:
        return list(explicit_columns)

    metadata = artifacts.get("metadata") or {}
    for key in ("feature_columns", "best_selected_features", "selected_features"):
        columns = metadata.get(key) or artifacts.get(key)
        if columns:
            artifacts["feature_columns"] = list(columns)
            return artifacts["feature_columns"]

    scaler = artifacts.get("scaler")
    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if scaler_feature_count == len(ALL_ENGINEERED_FEATURE_COLUMNS):
        inferred_columns = ALL_ENGINEERED_FEATURE_COLUMNS
    else:
        inferred_columns = FEATURE_COLUMNS

    artifacts["feature_columns"] = list(inferred_columns)
    return artifacts["feature_columns"]


def _normalize_prediction_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    feature_columns = _infer_feature_columns(artifacts)
    scaler = artifacts["scaler"]
    scaler_feature_count = getattr(scaler, "n_features_in_", len(feature_columns))

    if scaler_feature_count == len(ALL_ENGINEERED_FEATURE_COLUMNS) and len(feature_columns) != scaler_feature_count:
        artifacts["scaler"] = _reduce_scaler_to_features(
            scaler=scaler,
            source_columns=ALL_ENGINEERED_FEATURE_COLUMNS,
            target_columns=feature_columns,
        )
        return artifacts

    if scaler_feature_count != len(feature_columns):
        raise ValueError(
            "Artifact feature configuration is inconsistent: "
            f"scaler expects {scaler_feature_count} engineered features, "
            f"but the selected feature list has {len(feature_columns)} columns."
        )

    return artifacts


def _expected_feature_count(artifacts: dict[str, Any]) -> int:
    word_vectorizer = artifacts["word_vectorizer"]
    vocabulary = getattr(word_vectorizer, "vocabulary_", None) or {}
    feature_columns = artifacts["feature_columns"]
    return len(vocabulary) + len(feature_columns)


def _validate_artifact_feature_contract(artifacts: dict[str, Any]) -> None:
    metadata = artifacts.get("metadata") or {}
    model = artifacts["model"]
    expected_feature_count = _expected_feature_count(artifacts)
    model_feature_count = getattr(model, "n_features_in_", None)

    if metadata.get("uses_char_vectorizer") is True:
        raise ValueError(
            "The saved artifacts still expect character TF-IDF features, but the "
            "current pipeline has removed the char vectorizer. Retrain and resave "
            "the artifacts with the latest src/toxic_pipeline.py pipeline."
        )

    if model_feature_count is not None and model_feature_count != expected_feature_count:
        raise ValueError(
            "The saved model feature shape does not match the current word-only "
            f"pipeline (model expects {model_feature_count}, pipeline builds "
            f"{expected_feature_count}). Retrain and resave the artifacts."
        )


def vectorize_comment(comment: str, artifacts: dict[str, Any]) -> Any:
    artifacts = _normalize_prediction_artifacts(artifacts)
    raw_text = str(comment)
    protected_text = protect_non_toxic_negations(raw_text)
    clean = clean_text(protected_text)
    engineered = build_all_engineered_features(raw_text)[artifacts["feature_columns"]]
    scaler = artifacts["scaler"]

    word_vectorizer = artifacts["word_vectorizer"]

    x_word = word_vectorizer.transform([clean])
    x_eng = csr_matrix(scaler.transform(engineered.values))
    return hstack([x_word, x_eng], format="csr")


def _predict_from_features(features: Any, model: Any) -> dict[str, Any]:
    label = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return {
        "label": "Toxic" if label == 1 else "Not Toxic",
        "probability": probability,
    }


def predict_toxicity_raw_model(comment: str, artifacts: dict[str, Any]) -> dict[str, Any]:
    model = artifacts["model"]
    features = vectorize_comment(comment, artifacts)
    return _predict_from_features(features, model)


def predict_toxicity(comment: str, artifacts: dict[str, Any]) -> dict[str, Any]:
    model = artifacts["model"]
    if should_override_short_neutral_input(comment):
        return {
            "label": "Not Toxic",
            "probability": 0.15,
        }

    features = vectorize_comment(comment, artifacts)
    return _predict_from_features(features, model)
