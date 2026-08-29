#!/usr/bin/env bash
set -u
src="dataset/new_benchmarks/autodl_forecast_commands_remote_code.sh"
skip_file="dataset/new_benchmarks/local_existing_cache_paths.txt"
while IFS= read -r cmd; do
  if [[ "$cmd" != python* && "$cmd" != /root/miniconda3/bin/python* ]]; then
    continue
  fi
  cache_path=$(printf "%s\n" "$cmd" | sed -n "s/.*--cache_save_path \([^ ]*\).*/\1/p")
  if [ -n "$cache_path" ] && [ -f "$cache_path" ]; then
    echo "[SKIP] $cache_path"
    continue
  fi
  if [ -n "$cache_path" ] && [ -f "$skip_file" ] && grep -Fxq "$cache_path" "$skip_file"; then
    echo "[SKIP_LOCAL] $cache_path"
    continue
  fi
  echo "[RUN] $cache_path"
  eval "$cmd"
  status=$?
  if [ $status -eq 0 ]; then
    rm -rf /root/autodl-tmp/checkpoints_new
    echo "[CLEAN] /root/autodl-tmp/checkpoints_new"
  fi
done < "$src"
