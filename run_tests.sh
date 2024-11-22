#!/bin/bash

# /run_tests.sh

# Usage:
#   ./run_tests.sh                 # Run all tests
#   ./run_tests.sh <test_folder>   # Run tests in a specific folder

if [ "$#" -eq 0 ]; then
    # Run all tests
    pytest
elif [ "$#" -eq 1 ]; then
    # Run tests in the specified folder
    pytest "$1"
else
    echo "Usage: $0 [test_folder]"
    exit 1
fi