#!/bin/bash
if [ -d ./bin ]; then 
    if [ -n "$1" ]; then
        ./bin/ESN $1 2>err.log
    else 
        echo "\nYou have to set data file\n"
    fi
else 
    echo "\nYou have to build first\n"
fi