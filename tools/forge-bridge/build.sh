#!/usr/bin/env bash
# Compile the Forge bridge against a local Forge install.
# Usage: FORGE_DIR=/path/to/forge tools/forge-bridge/build.sh
set -euo pipefail
: "${FORGE_DIR:?set FORGE_DIR to the unpacked Forge 2.0.15 directory}"
JAVA_HOME="${JAVA_HOME:-}"
JAVAC="${JAVA_HOME:+$JAVA_HOME/bin/}javac"
HERE="$(cd "$(dirname "$0")" && pwd)"
JAR="$(ls "$FORGE_DIR"/forge-gui-desktop-*-jar-with-dependencies.jar | head -1)"
mkdir -p "$HERE/out"
"$JAVAC" -cp "$JAR" -d "$HERE/out" $(find "$HERE/src" -name '*.java')
echo "built against $JAR"
