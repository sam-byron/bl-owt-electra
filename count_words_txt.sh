#!/usr/bin/env bash
# filepath: count_txt_words.sh

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <txt_folder>" >&2
  exit 1
fi

TXT_ROOT="$1"

if [ ! -d "$TXT_ROOT" ]; then
  echo "ERROR: Directory '$TXT_ROOT' not found." >&2
  exit 1
fi

total=0
echo "Word counts per file:"

# Use process substitution so the loop runs in this shell, not a subshell
while IFS= read -r -d '' file; do
  words=$(wc -w < "$file")
  printf "%8d  %s\n" "$words" "$file"
  total=$(( total + words ))
done < <(find "$TXT_ROOT" -type f -name '*.txt' -print0)

echo
echo "Total words across all files: $total"
