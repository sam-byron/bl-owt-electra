#!/usr/bin/env bash
# filepath: parse_check.sh

set -euo pipefail

# --- sanity checks ---
for cmd in find xmllint xmlstarlet bc wc awk parallel; do
  if ! command -v $cmd &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <xml_root_dir> <txt_output_dir> [num_jobs]" >&2
  exit 1
fi

XML_ROOT="$1"
TXT_ROOT="$2"
NUM_JOBS="${3:-$(nproc)}"  # Default to number of CPU cores

if [ ! -d "$XML_ROOT" ]; then
  echo "ERROR: XML directory '$XML_ROOT' not found." >&2
  exit 1
fi

mkdir -p "$TXT_ROOT"

# Function to process a single XML file
process_xml() {
  local xml="$1"
  local xml_root="$2"
  local txt_root="$3"
  
  local rel="${xml#$xml_root/}"
  local out="$txt_root/${rel%.xml}.txt"
  mkdir -p "$(dirname "$out")"
  
  if xmllint --recover "$xml" 2>/dev/null \
     | xmlstarlet sel -t -m "//p" -v "normalize-space(string(.))" -n \
       > "$out"; then
    echo "✅ $rel"
  else
    echo "❌ $rel" >&2
    return 1
  fi
}

# Export function and variables for parallel
export -f process_xml
export XML_ROOT TXT_ROOT

# --- count xml files ---
xml_files_count=$(find "$XML_ROOT" -type f -iname '*.xml' | wc -l)
echo "DEBUG: found $xml_files_count XML files under '$XML_ROOT'."
echo "DEBUG: using $NUM_JOBS parallel jobs."

if [ "$xml_files_count" -eq 0 ]; then
  echo "ERROR: no .xml files found — check your XML_ROOT path!" >&2
  exit 1
fi

# --- conversion loop with parallel processing ---
echo
echo "1) Converting XML → TXT with $NUM_JOBS parallel jobs…"

find "$XML_ROOT" -type f -iname '*.xml' -print0 | \
  parallel -0 -j "$NUM_JOBS" --progress process_xml {} "$XML_ROOT" "$TXT_ROOT"

echo
echo "DEBUG: parallel conversion finished."