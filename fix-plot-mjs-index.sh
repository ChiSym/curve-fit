#!/bin/env bash

f="node_modules/@use-gpu/plot/mjs/index.mjs"

if [ ! -f "$f" ]; then
	echo "File not found: $f"
	exit 1
fi

sed -i.bak 's|export \* from '\''\./util\.mjs'\'';|export \* from '\''\./util/index\.mjs'\'';|' "$f"
rm "$f.bak"

echo "$f was updated"
