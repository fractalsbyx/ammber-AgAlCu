#!/usr/bin/env bash
# Download the Ag-Al-Cu thermodynamic database (Witusiewicz et al., J. Alloys
# Compd. 387 (2005) 217) into the repository root. It is not distributed with
# this repository. The download is pinned to one commit of
# PhasesResearchLab/MaterialsMap and checked against a SHA-256 checksum.
set -euo pipefail

COMMIT=83ab2eb67b1ce0af5b9e9fc7adf83816acf02bf3
URL="https://raw.githubusercontent.com/PhasesResearchLab/MaterialsMap/${COMMIT}/tasks/Ag-Al-Cu.TDB"
SHA256=bd5ed0092c11f23176f9e44a562bb7fe9f207060a79dfb79e63f32d991b1ef9d
DEST="$(cd "$(dirname "$0")" && pwd)/Ag-Al-Cu.TDB"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [ -f "$DEST" ] && [ "$(sha256 "$DEST")" = "$SHA256" ]; then
  echo "Ag-Al-Cu.TDB already present and verified"
  exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$URL" -o "$tmp"
if [ "$(sha256 "$tmp")" != "$SHA256" ]; then
  echo "Checksum mismatch for $URL; not installing it" >&2
  exit 1
fi
mv "$tmp" "$DEST"
trap - EXIT
echo "Downloaded and verified $DEST"
