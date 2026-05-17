#!/usr/bin/env bash

# Normalise common Windows tool locations for Git Bash without affecting
# Linux/macOS. This lets the Bash scripts find Docker and winget-installed tools
# when the student runs them from Git Bash during local verification.
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* || -n "${WINDIR:-}" ]]; then
  to_unix_path() {
    local raw_path="$1"
    if command -v cygpath >/dev/null 2>&1; then
      cygpath -u "${raw_path}" 2>/dev/null || printf '%s\n' "${raw_path}"
    else
      printf '%s\n' "${raw_path}"
    fi
  }

  add_path_if_dir() {
    local candidate="$1"
    if [[ -d "${candidate}" ]]; then
      case ":${PATH}:" in
        *":${candidate}:"*) ;;
        *) export PATH="${PATH}:${candidate}" ;;
      esac
    fi
  }

  add_path_if_dir "$(to_unix_path 'C:\Program Files\Docker\Docker\resources\bin')"

  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    add_path_if_dir "$(to_unix_path "${LOCALAPPDATA}\\Microsoft\\WinGet\\Links")"
  fi

  if [[ -n "${USERPROFILE:-}" ]]; then
    add_path_if_dir "$(to_unix_path "${USERPROFILE}\\AppData\\Local\\Microsoft\\WinGet\\Links")"
  fi
fi
