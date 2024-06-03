#!/bin/sh
#
# pre-commit runner for eslint. This is required to deal with the fact that Node
# builds a `module.paths` that looks for `node_modules` in the current directory
# and all parents, NOT in `$NODE_PATH`.
#
# We hack this with a symlink.

symlink_created=false

function cleanup {
  if [ "$symlink_created" = true ]; then
    rm -rf ./node_modules
    echo "Cleaned up temporary node_modules path."
  fi
}

trap cleanup EXIT

if [ ! -d ./node_modules ]; then
  echo "Creating node_modules symlink"

  ln -s $NODE_PATH .
  symlink_created=true
else
  echo "../node_modules already exists. Skipping symlink creation."
fi

eslint
