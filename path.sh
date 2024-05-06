
# if PYTHONPATH is not set, set it to the current directory
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$PWD
fi