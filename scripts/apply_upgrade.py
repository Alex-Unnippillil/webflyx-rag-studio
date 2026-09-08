from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path


def safe_extract(archive: tarfile.TarFile, root: Path) -> None:
    root = root.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    archive.extractall(root)


def main() -> None:
    root = Path.cwd().resolve()
    part_dir = root / ".upgrade"
    parts = sorted(part_dir.glob("part-*"))
    if len(parts) < 10:
        raise SystemExit(f"Upgrade payload incomplete: found {len(parts)} chunks.")

    payload = b"".join(part.read_bytes() for part in parts)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        safe_extract(archive, root)

    shutil.rmtree(part_dir, ignore_errors=True)
    (root / "scripts" / "apply_upgrade.py").unlink(missing_ok=True)
    (root / ".github" / "workflows" / "apply-v1.2.yml").unlink(missing_ok=True)
    print("Applied Webflyx RAG Studio 1.2.0 upgrade.")


if __name__ == "__main__":
    main()
