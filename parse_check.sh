#!/usr/bin/env bash
# filepath: parse_check.sh

set -euo pipefail

# --- sanity checks ---
for cmd in find xmllint xmlstarlet bc wc awk; do
  if ! command -v $cmd &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <xml_root_dir> <txt_output_dir>" >&2
  exit 1
fi

XML_ROOT="$1"
TXT_ROOT="$2"

if [ ! -d "$XML_ROOT" ]; then
  echo "ERROR: XML directory '$XML_ROOT' not found." >&2
  exit 1
fi

mkdir -p "$TXT_ROOT"

# --- count xml files ---
xml_files_count=$(find "$XML_ROOT" -type f -iname '*.xml' | wc -l)
echo "DEBUG: found $xml_files_count XML files under '$XML_ROOT'."

if [ "$xml_files_count" -eq 0 ]; then
  echo "ERROR: no .xml files found — check your XML_ROOT path!" >&2
  exit 1
fi

# --- conversion loop ---
echo
echo "1) Converting XML → TXT…"
processed=0
find "$XML_ROOT" -type f -iname '*.xml' -print0 |
while IFS= read -r -d '' xml; do
  processed=$((processed+1))
  rel="${xml#$XML_ROOT/}"
  out="$TXT_ROOT/${rel%.xml}.txt"
  mkdir -p "$(dirname "$out")"

  printf "  [%4d/%d] %s → " "$processed" "$xml_files_count" "$rel"

  if xmllint --recover "$xml" 2>/dev/null \
     | xmlstarlet sel -t -m "//p" -v "normalize-space(string(.))" -n \
       > "$out"; then
    echo "ok"
  else
    echo "⚠️  failed"
  fi
done

echo
echo "DEBUG: conversion loop finished — processed $processed files."

# --- count <w> in XML ---
echo
echo "2) Counting <w> tokens in XML…"
xml_count=$(find "$XML_ROOT" -type f -iname '*.xml' -print0 \
  | xargs -0 xmlstarlet sel -t -v "count(//w)" -n \
  | paste -sd+ - | bc)
echo "   XML <w> count: $xml_count"

# --- count words in TXT ---
echo
echo "3) Counting whitespace‐split words in TXT…"
txt_count=$(find "$TXT_ROOT" -type f -name '*.txt' -print0 \
  | xargs -0 wc -w \
  | tail -n1 | awk '{print $1}')
echo "   TXT word count: $txt_count"

# --- final check ---
echo
if [ "$xml_count" -eq "$txt_count" ]; then
  echo "✅ Success: XML and TXT word counts match!"
else
  echo "⚠️  Mismatch: dropped $((xml_count - txt_count)) words during conversion."
  exit 1
fi
