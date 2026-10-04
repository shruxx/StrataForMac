#!/bin/sh
# Strata for macOS & Linux: the first run installs everything and starts the model; later runs just start it.
cd "$(dirname "$0")" || exit 1

if [ "$(uname -s)" = "Darwin" ]; then
  export DEVELOPER_DIR="/Library/Developer/CommandLineTools"
  export PATH="$PWD/.venv/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
fi


# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then
      PY=$c; break
    fi
  done
  if [ -z "$PY" ] && command -v uv >/dev/null 2>&1; then
    for v in 3.13 3.12 3.11 3.10; do
      UVPY=$(uv python find $v 2>/dev/null)
      if [ -n "$UVPY" ] && ok_py "$UVPY"; then
        PY="$UVPY"; break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it ..."
    if [ "$(uname -s)" = "Darwin" ]; then
      if command -v brew >/dev/null 2>&1; then
        brew install python
        PY=python3
      else
        echo "Please install Python 3.10 or newer (or Homebrew: https://brew.sh), then run ./setup.sh again."
        exit 1
      fi
    elif command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
      PY=python3
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
      PY=python3
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
      PY=python3
    fi
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv, then run ./setup.sh again."
      exit 1
    fi
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv"; exit 1; }
fi
export PATH="$PWD/.venv/bin:$PATH"
exec .venv/bin/python setup.py "$@"
