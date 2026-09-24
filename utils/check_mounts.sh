#!/usr/bin/env bash
# Warn (never fail) when any of the SSD_COUNT paths named by configs.h's
# SSD_ROOT is missing or is a plain directory rather than a mount point.  An
# unmounted /mnt/ssdN silently sends every write to the parent filesystem
# (usually /): benchmarks then measure the wrong device and a large run can
# fill the root disk.
#
# SSD_COUNT / SSD_ROOT are read straight from configs.h so this can't drift
# from the C++ side.  Set PLAID_SKIP_MOUNT_CHECK=1 to silence (e.g. a
# deliberate tmpfs dev box).  Always exits 0.

[ -n "${PLAID_SKIP_MOUNT_CHECK:-}" ] && [ "$PLAID_SKIP_MOUNT_CHECK" != 0 ] && exit 0

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cfg="$repo/configs.h"

ssd_count=$(sed -n 's/^constexpr size_t SSD_COUNT *= *\([0-9]*\);.*/\1/p' "$cfg")
ssd_root=$(sed -n 's/^const std::string SSD_ROOT *= *"\([^"]*\)";.*/\1/p' "$cfg")
if [ -z "$ssd_count" ] || [ -z "$ssd_root" ]; then
  echo "WARNING: mount check: could not parse SSD_COUNT/SSD_ROOT from $cfg; skipping" >&2
  exit 0
fi

have_mountpoint=0
command -v mountpoint >/dev/null 2>&1 && have_mountpoint=1

# is_mount <dir>: mountpoint(1) if available, else "different st_dev than the
# parent" (misses a bind mount of the same filesystem, good enough as a fallback).
is_mount() {
  if [ $have_mountpoint -eq 1 ]; then
    mountpoint -q "$1"
  else
    [ "$(stat -c %d "$1")" != "$(stat -c %d "$1/..")" ]
  fi
}

# Status per index, then collapse consecutive identical statuses to ranges.
bad=0
lines=()
run_start=-1 run_status="" run_path=""
flush() {
  [ $run_start -lt 0 ] && return
  [ "$run_status" = ok ] && return
  local label
  if [ $run_start -eq $1 ]; then
    label="$(printf "$ssd_root" "$run_start")"
  else
    label="$(printf "$ssd_root" "$run_start") .. $(printf "$ssd_root" "$1")"
  fi
  lines+=("  $label  $run_status")
}
for ((i = 0; i < ssd_count; i++)); do
  p="$(printf "$ssd_root" "$i")"
  if [ ! -d "$p" ]; then
    st="missing"
  elif ! is_mount "$p"; then
    parent_fs="$(df --output=target "$p" 2>/dev/null | tail -n 1)"
    st="not a mount (writes land on ${parent_fs:-the parent filesystem})"
  else
    st="ok"
  fi
  [ "$st" != ok ] && bad=$((bad + 1))
  if [ "$st" != "$run_status" ]; then
    flush $((i - 1))
    run_start=$i run_status="$st"
  fi
done
flush $((ssd_count - 1))

if [ $bad -eq 0 ]; then
  echo "mount check: $ssd_count/$ssd_count SSD paths are mounts"
else
  {
    echo "WARNING: $bad of $ssd_count SSD paths are not mount points (configs.h SSD_ROOT=$ssd_root):"
    printf '%s\n' "${lines[@]}"
    echo "  Set PLAID_SKIP_MOUNT_CHECK=1 to silence (e.g. a deliberate tmpfs dev box)."
  } >&2
fi
exit 0
