#!/bin/bash
set -xe

git init bamboo
cd bamboo
git remote add origin https://github.com/bytedance/bamboo.git
git fetch --depth=1 origin 21ce529aa4825cc912a36c01b36a111693a1b451
git checkout FETCH_HEAD
git apply ../plugin.patch
