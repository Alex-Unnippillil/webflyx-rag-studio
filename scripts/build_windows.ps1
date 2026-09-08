$ErrorActionPreference = "Stop"

uv sync --all-groups

uv run ruff check .
uv run pytest -q

uv run python scripts/make_icon.py

Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

uv run pyinstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name WebflyxRAG `
    --icon assets/webflyx.ico `
    --collect-all sentence_transformers `
    --collect-all transformers `
    --collect-all keyring `
    desktop_app.py

Write-Host ""
Write-Host "Portable application:"
Write-Host "dist\WebflyxRAG\WebflyxRAG.exe"
