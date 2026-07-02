#! /usr/bin/bash
# extract performance numbers from raw fio results
for file in "$@"; do
  echo "$file"
  grep "READ" "$file" | sed -e 's/^.*bw=[^(]\+(\([^)]\+\)),.*/\1/'
done
