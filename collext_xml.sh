#!/usr/bin/env bash
# Usage: collect_xml.sh <source_dir> <dest_dir>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_dir> <dest_dir>"
  exit 1
fi

src="$1"
dst="$2"

# make sure source exists
if [ ! -d "$src" ]; then
  echo "Error: source directory '$src' does not exist."
  exit 1
fi

# create destination if needed
mkdir -p "$dst"

# find and copy
find "$src" -type f -name '*.xml' -print0 | while IFS= read -r -d '' file; do
  cp -v "$file" "$dst"
done

echo "All .xml files copied from '$src' to '$dst'."