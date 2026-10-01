#!/usr/bin/env bash
set -euo pipefail

# Install the package beside this script, regardless of the calling directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 -m pip install -e .
