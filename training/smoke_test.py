from pathlib import Path
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "run_config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project = config.get("project", {})
    model = config.get("model", {})
    dataset = config.get("dataset", {})

    print("Phase 1 smoke test passed.")
    print(f"Project: {project.get('name')}")
    print(f"Base model: {model.get('base_model')}")
    print(f"Expected train file: {dataset.get('train_file')}")


if __name__ == "__main__":
    main()
