#!/usr/bin/env bash
# filepath: convert_all_xml_to_txt.sh

set -euo pipefail

# requirements: xmlstarlet, xmllint, wc
for cmd in xmlstarlet xmllint wc; do
  if ! command -v $cmd &> /dev/null; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <xml_root_dir> [txt_output_dir]" >&2
  exit 1
fi

XML_ROOT="$1"
TXT_ROOT="${2:-$XML_ROOT-txt}"

mkdir -p "$TXT_ROOT"

total_words=0

# recurse into all .xml under XML_ROOT
find "$XML_ROOT" -type f -name '*.xml' -print0 |
while IFS= read -r -d '' xml; do
  # derive parallel .txt path
  rel="${xml#$XML_ROOT/}"
  txt="$TXT_ROOT/${rel%.xml}.txt"
  mkdir -p "$(dirname "$txt")"

  # recover & stream through xmlstarlet
  xmllint --recover "$xml" 2>/dev/null |
    # xmlstarlet sel -t \
    #   -m "//s" \
    #     -m ".//w|.//c" \
    #       -v "normalize-space(.)" \
    #       -o " " \
    #     -b \
    #     -n \
    # xmlstarlet sel -t \
    # -m "//s" \
    #     -v "normalize-space(string(.))" \
    #     -n \
    xmlstarlet sel -t -m "//s" -v "normalize-space(string(.))" -n \
    > "$txt"

  # count words in this txt, add to total
  words=$(wc -w < "$txt")
  (( total_words += words ))

  echo "Converted: $xml → $txt ($words words)"
done

echo
echo "=== Conversion complete — Total words = $total_words ==="
