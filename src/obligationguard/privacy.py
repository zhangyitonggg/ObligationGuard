from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re


HOME_PATH = re.compile(r"(?:(?i:[A-Za-z]:[\\/]+Users[\\/]+)|/ho" + r"me/|/Use" + r"rs/)[^\\/\s\"'`<>:;,\[\](){}]+")
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+\\-])[A-Za-z0-9._%+-]{1,64}@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:sk-proj-|sk-ant-api)[A-Za-z0-9_-]{16,}|"
    r"(?:hvs|hvr|hvb)\.[A-Za-z0-9_-]{12,}|"
    r"[sr]k_(?:live|test)_[A-Za-z0-9]{12,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|SK[a-fA-F0-9]{32}|key-[a-fA-F0-9]{32}"
)
PRIVATE_KEY = re.compile(r"-----BEGIN ((?:RSA |EC |OPENSSH )?PRIVATE KEY)-----.*?-----END \1-----", re.DOTALL)
PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net", "example.invalid", "localhost.invalid"}


def placeholder_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(domain == item or domain.endswith("." + item) for item in PLACEHOLDER_DOMAINS)


def remove_private_text(value: str, identity_terms: tuple[str, ...] = ()) -> str:
    def email(match):
        return match.group(0) if placeholder_domain(match.group(1)) else ""
    value = EMAIL.sub(email, value)
    value = HOME_PATH.sub("", value)
    value = KEY.sub("", PRIVATE_KEY.sub("", value))
    for term in sorted(identity_terms, key=len, reverse=True):
        if term:
            value = re.sub(re.escape(term), "", value, flags=re.IGNORECASE)
    return value


def remove_private_record(value, identity_terms: tuple[str, ...] = ()):
    if isinstance(value, str):
        return remove_private_text(value, identity_terms)
    if isinstance(value, dict):
        return {key: remove_private_record(item, identity_terms) for key, item in value.items() if remove_private_text(key, identity_terms) == key}
    if isinstance(value, list):
        return [remove_private_record(item, identity_terms) for item in value]
    return value


def inspect_text(text: str, identity_terms: tuple[str, ...] = ()) -> Counter:
    findings = Counter()
    findings["personal_paths"] = len(HOME_PATH.findall(text))
    findings["contact_addresses"] = sum(not placeholder_domain(domain) for domain in EMAIL.findall(text))
    findings["credentials"] = len(KEY.findall(text))
    for term in identity_terms:
        if term and re.search(re.escape(term), text, re.IGNORECASE):
            findings["identity_terms"] += 1
    return +findings


def inspect_pdf(path: Path, identity_terms: tuple[str, ...] = ()) -> Counter:
    """Inspect visible vector text and PDF metadata; reject uninspected binary content."""
    try:
        import pypdfium2
        from pypdf import PdfReader
    except ImportError:
        return Counter({"uninspected_pdf_files": 1})
    findings = Counter()
    document = pypdfium2.PdfDocument(str(path))
    try:
        for page in document:
            text_page = page.get_textpage()
            try:
                findings.update(inspect_text(text_page.get_text_range(), identity_terms))
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    reader = PdfReader(path)
    findings.update(inspect_text(str(reader.metadata or {}), identity_terms))
    if reader.xmp_metadata:
        findings.update(inspect_text(reader.xmp_metadata.stream.get_data().decode("utf-8"), identity_terms))
    for page in reader.pages:
        findings["uninspected_pdf_images"] += len(page.images)
        for annotation in page.get("/Annots", []):
            findings.update(inspect_text(str(annotation.get_object()), identity_terms))
    findings["uninspected_pdf_attachments"] += len(reader.attachments)
    return +findings


def inspect_image(path: Path, reviewed_sha256: str | None, identity_terms: tuple[str, ...] = ()) -> Counter:
    from .io import file_sha256
    if not reviewed_sha256 or file_sha256(path) != reviewed_sha256:
        return Counter({"unreviewed_image_files": 1})
    try:
        from PIL import Image
    except ImportError:
        return Counter({"uninspected_image_metadata": 1})
    findings = Counter()
    with Image.open(path) as picture:
        picture.verify()
    with Image.open(path) as picture:
        findings.update(inspect_text(str(picture.info), identity_terms))
        findings.update(inspect_text(str(dict(picture.getexif())), identity_terms))
    return findings


def audit_repository(root: str | Path, identity_env: str | None = None) -> dict:
    root = Path(root).resolve()
    terms = tuple(x.strip() for x in os.getenv(identity_env, "").split(";") if x.strip()) if identity_env else ()
    totals = Counter()
    files = 0
    findings = []
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache"}
    image_sources = root / "docs" / "assets" / "sources.json"
    reviewed_images = json.loads(image_sources.read_text(encoding="utf-8")) if image_sources.is_file() else {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        result = inspect_text(relative, terms)
        if path.suffix.lower() == ".pdf":
            result.update(inspect_pdf(path, terms))
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            source = reviewed_images.get(relative, {})
            result.update(inspect_image(path, source.get("sha256") if source.get("visually_reviewed") is True else None, terms))
        else:
            try:
                with path.open(encoding="utf-8-sig") as stream:
                    for line in stream:
                        result.update(inspect_text(line, terms))
            except UnicodeError:
                result["uninspected_binary_files"] += 1
        files += 1
        totals.update(result)
        if result:
            findings.append({"file": relative if not inspect_text(relative, terms) else "<redacted filename>", "counts": dict(result)})
    return {"files_checked": files, "passed": not totals, "counts": dict(totals), "findings": findings}
