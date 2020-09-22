#!/bin/sh
ln -sf ./.githooks hooks
rm -r .git/hooks
mv -f hooks .git/
conda install -n base black
conda config --set auto_stack
conda env create --force --file environment.yml
