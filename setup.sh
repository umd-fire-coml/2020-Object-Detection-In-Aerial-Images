#!/bin/bash
if [[ $(uname -s) == MINGW64* ]]; then
    export MSYS=winsymlinks:nativestrict
fi
rm -r .git/hooks
ln -sf ../.githooks .git/hooks
conda install -n base black
conda config --set auto_stack 1
conda env create --force --file environment.yml
