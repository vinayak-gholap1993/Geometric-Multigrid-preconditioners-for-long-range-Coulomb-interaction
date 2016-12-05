#!/bin/bash
#rm Log.txt
cmake .
make runclean
make

make run >> "Log.output"

