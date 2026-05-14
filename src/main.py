from __future__ import annotations

import sys

from toxic_pipeline import load_best_artifacts, predict_toxicity_raw_model


def load_artifacts_or_exit() -> dict:
    try:
        return load_best_artifacts()
    except ValueError as exc:
        print(f"Artifact error: {exc}")
        print("Retrain and resave the artifacts with the latest pipeline, then try again.")
        raise SystemExit(1) from exc


def classify_once(comment: str) -> None:
    artifacts = load_artifacts_or_exit()
    result = predict_toxicity_raw_model(comment, artifacts)
    print(f"Prediction : {result['label']}")
    print(f"Toxic score: {result['probability']:.2%}")


def interactive_loop() -> None:
    artifacts = load_artifacts_or_exit()
    model_path = artifacts["paths"]["model"]
    print(f"Artifacts loaded from: {model_path.parent}")
    print("Type a comment to classify it. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            comment = input("Comment: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break

        if comment.lower() in {"exit", "quit"}:
            print("Stopped.")
            break
        if not comment:
            print("Please enter a non-empty comment.\n")
            continue

        result = predict_toxicity_raw_model(comment, artifacts)
        print(f"Prediction : {result['label']}")
        print(f"Toxic score: {result['probability']:.2%}\n")


def main() -> None:
    if len(sys.argv) > 1:
        classify_once(" ".join(sys.argv[1:]))
        return
    interactive_loop()


if __name__ == "__main__":
    main()
