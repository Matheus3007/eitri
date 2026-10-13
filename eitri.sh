#!/usr/bin/env bash
# Runs eitri with the venv on the Linux filesystem (the repo lives on /mnt/c, where venv I/O is ~10x slower).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/eitri}"
exec uv run --project "$(dirname "$(readlink -f "$0")")" --quiet "$@"
