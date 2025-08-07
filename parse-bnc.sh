#!/usr/bin/env bash
# filepath: convert_all_xml_to_txt.sh

set -euo pipefail

if ! command -v xmlstarlet &> /dev/null; then
  echo "ERROR: xmlstarlet is not installed. Install it via 'sudo apt install xmlstarlet' or similar." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <xml_root_dir> [txt_output_dir]" >&2
  exit 1
fi

XML_ROOT="$1"
TXT_ROOT="${2:-$XML_ROOT}"

# ensure output root exists
mkdir -p "$TXT_ROOT"

# find each .xml and convert
find "$XML_ROOT" -type f -name '*.xml' -print0 | while IFS= read -r -d '' XML; do
  # compute relative path and target txt path
  REL="${XML#$XML_ROOT/}"
  TXT="$TXT_ROOT/${REL%.xml}.txt"
  mkdir -p "$(dirname "$TXT")"

  # extract all text nodes, one per line
  # xmlstarlet sel -t -m "//text()" -v .  "$XML" > "$TXT"
  # xmlstarlet sel -t -m "//s" -m ".//w|.//c" -v "normalize-space(.)" -o " " -b -n "$XML" > "$TXT"
  # xmlstarlet sel -t -v "count(//w)" "$XML" > "$TXT"
  # xmlstarlet sel -t -m "//s" -m ".//w|.//c" -v "normalize-space(.)" -o " " -b -n "$XML" > "$TXT"
  xmlstarlet sel -t -m "//s" -v "normalize-space(string(.))" -n "$XML" > "$TXT"
  echo "Converted: $XML → $TXT"
done