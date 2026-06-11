#!/bin/bash
# Release gate for triplet-extract. Enforces the invariants that prevent
# stale-artifact uploads (a wheel built at one commit shipped after the
# tree moved on — versions are immutable on PyPI, so this class of error
# cannot be fixed in place, only superseded).
#
# Invariants enforced, in order:
#   1. Working tree is clean and HEAD is pushed.
#   2. pyproject version == triplet_extract.__version__ == vX.Y.Z tag,
#      and that tag points at HEAD. Tags are NEVER re-pointed: a new tree
#      state gets a new version, always.
#   3. The version does not already exist on PyPI.
#   4. dist/ is rebuilt from scratch at HEAD (never reused).
#   5. The BUILT WHEEL (not the source tree) passes behavioral smoke
#      probes in a throwaway venv — import, version match, and one probe
#      per recent soundness class (negation polarity, asserter chains,
#      quote attribution).
#   6. Only then: twine upload, with explicit confirmation.
#
# NOTE: the PRIMARY release path is CI trusted publishing — push a
# vX.Y.Z tag and .github/workflows/release.yml builds, verifies, and
# publishes from that tag's tree via OIDC. This script is the local
# pre-flight (--check) and a manual fallback.
#
# Usage: ./release.sh            (gate + upload prompt)
#        ./release.sh --check    (gate only, no upload)
set -euo pipefail

echo "== 1/6 tree state =="
test -z "$(git status --porcelain)" || { echo "FAIL: working tree not clean"; exit 1; }
git fetch -q origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" \
  || { echo "FAIL: HEAD not pushed to origin/main"; exit 1; }

echo "== 2/6 version coherence =="
VERSION=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)"/\1/')
MODVERSION=$(grep -m1 '__version__' triplet_extract/__init__.py | sed 's/.*"\(.*\)"/\1/')
test "$VERSION" = "$MODVERSION" \
  || { echo "FAIL: pyproject=$VERSION but __version__=$MODVERSION"; exit 1; }
TAG="v$VERSION"
test "$(git rev-parse "$TAG^{commit}" 2>/dev/null)" = "$(git rev-parse HEAD)" \
  || { echo "FAIL: tag $TAG missing or not at HEAD (never re-point: bump instead)"; exit 1; }

echo "== 3/6 version unused on PyPI =="
if curl -sf "https://pypi.org/pypi/triplet-extract/$VERSION/json" > /dev/null; then
  echo "FAIL: $VERSION already on PyPI — bump the version"; exit 1
fi

echo "== 4/6 fresh build at HEAD =="
rm -rf dist build
uv build
uvx twine check dist/*

echo "== 5/6 smoke-test the built wheel (throwaway venv) =="
SMOKE=$(mktemp -d)
trap 'rm -rf "$SMOKE"' EXIT
uv venv -q --python 3.11 "$SMOKE/venv"
SM_WHEEL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
uv pip install -q --python "$SMOKE/venv/bin/python" dist/*.whl "en-core-web-sm @ $SM_WHEEL"
REPO_DIR=$(pwd)
cd "$SMOKE"  # ensure the INSTALLED package is imported, not the repo tree
"$SMOKE/venv/bin/python" "$REPO_DIR/.github/workflows/smoke_probes.py" "$VERSION"
cd - > /dev/null

echo "== 6/6 upload =="
if [ "${1:-}" = "--check" ]; then
  echo "CHECK PASSED for $VERSION (no upload; rerun without --check to publish)"
  exit 0
fi
echo "About to upload triplet-extract $VERSION (commit $(git rev-parse --short HEAD)) to PyPI."
read -r -p "Type the version to confirm: " CONFIRM
test "$CONFIRM" = "$VERSION" || { echo "aborted"; exit 1; }
uvx twine upload dist/*
echo "Published $VERSION."
