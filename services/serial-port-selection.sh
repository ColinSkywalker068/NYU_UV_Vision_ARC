#!/usr/bin/env bash
# Shared preflight resolution; the bridge resolves again on every reconnect.
cv_resolve_serial_port() {
  local requested="${1:-auto}" directory="${2:-/dev/serial/by-id}"
  local path name
  local -a candidates=()
  if [ "$requested" != auto ]; then
    name="${requested##*/}"
    if [ -e "$requested" ] || [ "${requested%/*}" != "$directory" ] ||
       [[ ! "$name" =~ ^usb-NYUSH_Vision_Comm_port_[[:alnum:]]+-if00$ ]]; then
      printf '%s\n' "$requested"
      return 0
    fi
  fi
  for path in "$directory"/usb-NYUSH_Vision_Comm_port_*-if00; do
    name="${path##*/}"
    if [[ "$name" =~ ^usb-NYUSH_Vision_Comm_port_[[:alnum:]]+-if00$ ]] &&
       [ -L "$path" ] && [ -c "$path" ]; then
      candidates+=("$path")
    fi
  done
  case "${#candidates[@]}" in
    0) printf 'No eligible NYUSH Vision Comm controller in %s\n' "$directory" >&2; return 1 ;;
    1) printf '%s\n' "${candidates[0]}" ;;
    *) printf 'Ambiguous Vision Comm controllers; set CV_STM_SERIAL_PORT explicitly: %s\n' "${candidates[*]}" >&2; return 2 ;;
  esac
}
