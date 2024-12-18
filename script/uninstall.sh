#!/bin/bash

printf 'Listing these files; not really deleting any of them.\n\n'

uname_s=$(uname -s)
if [ "$uname_s" = Linux ]; then
find \
    /usr/local/openmm/lib \
    /usr/local/lib/python3.*/dist-packages \
    \( -type f -o -type d \) \
    \( -name '*CallbackPyForce*' \)
elif [ "$uname_s" = Darwin ]; then
find \
    $(python3 -c 'import site; print(site.getsitepackages()[-1])') \
    \( -type f -o -type d \) \
    \( -name '*CallbackPyForce*' \)
fi
