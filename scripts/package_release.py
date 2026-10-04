"""Build an anonymous source-and-data archive without Git or machine-local files."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obligationguard.datasets import load_instances, validate_benchmark, validate_training_split
from obligationguard.io import file_sha256, write_json
from obligationguard.privacy import audit_repository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--in-place", action="store_true", help="update checksums and package the source checkout itself")
    parser.add_argument("--archive", type=Path, help="ZIP destination (defaults to the output folder name)")
    parser.add_argument("--identity-env")
    args = parser.parse_args()
    output, data = args.output.resolve(), args.data.resolve()
    if (output == ROOT and not args.in_place) or (output != ROOT and output.is_relative_to(ROOT)):
        raise ValueError("release output must be outside the source checkout")
    archive = args.archive.resolve() if args.archive else output.parent / (output.name + ".zip")
    if archive.is_relative_to(output):
        raise ValueError("archive destination must be outside the release directory")
    required_data = ("train.jsonl", "validation.jsonl", "manifest.json", "split_ids.json", "obligationbench.jsonl", "01_positive/obligationbench_positive.jsonl", "02_negative/obligationbench_negative.jsonl", "benchmark_manifest.json", "without_templates/train_42000.jsonl", "without_templates/source_manifest.json", "without_templates/train.jsonl", "without_templates/validation.jsonl", "without_templates/split_ids.json", "without_templates/manifest.json")
    missing = [name for name in required_data if not (data / name).is_file()]
    if missing:
        raise ValueError("required release data missing: " + ", ".join(missing))
    validate_training_split(load_instances(data / "train.jsonl"), load_instances(data / "validation.jsonl"))
    validate_training_split(load_instances(data / "without_templates/train.jsonl"), load_instances(data / "without_templates/validation.jsonl"))
    benchmark = data / "obligationbench.jsonl"
    instances = load_instances(benchmark)
    validate_benchmark(instances)
    for folder, split, positive in (("01_positive", "positive", True), ("02_negative", "negative", False)):
        actual = [item.to_dict() for item in load_instances(data / folder / ("obligationbench_" + split + ".jsonl"))]
        expected = [item.to_dict() for item in instances if bool(item.obligations) == positive]
        if actual != expected:
            raise ValueError("benchmark split does not match the complete dataset")
    selected = [ROOT / name for name in ("README.md", "pyproject.toml", ".gitattributes", ".gitignore", "Supplementary_Materials.pdf")]
    selected += sorted(ROOT.glob("requirements-*.txt"))
    for directory in ("configs", "docs", "scripts", "src", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".md", ".toml", ".json", ".txt", ".sh", ".bat", ".svg", ".png"} and ".local." not in path.name:
                selected.append(path)
    files = []
    for source in selected:
        if not source.resolve().is_relative_to(ROOT):
            raise ValueError("source file resolves outside the checkout")
        destination = output / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        files.append(destination)
    for name in required_data:
        source = data / name
        if source.is_file():
            destination = output / "data" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination.resolve():
                shutil.copyfile(source, destination)
            files.append(destination)
    report = audit_repository(output, args.identity_env)
    if not report["passed"]:
        raise ValueError("release privacy audit did not pass")
    checksum_file = output / "FILE_CHECKSUMS.json"
    write_json(checksum_file, {"sha256": {path.relative_to(output).as_posix(): file_sha256(path) for path in sorted(files)}})
    files.append(checksum_file)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(files):
            entry = zipfile.ZipInfo(output.name + "/" + path.relative_to(output).as_posix())
            entry.create_system = 3
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
            with path.open("rb") as source, bundle.open(entry, "w") as destination:
                shutil.copyfileobj(source, destination)
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("archive integrity verification failed")
    temporary.replace(archive)
    print(f"Packaged {len(files)} files; privacy audit passed.")


if __name__ == "__main__":
    main()
