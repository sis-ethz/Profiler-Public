export PROFILERHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Profiler home directory: $PROFILERHOME"
export PYTHONPATH="$PYTHONPATH:$PROFILERHOME"
export PATH="$PATH:$PROFILERHOME"
echo $PATH
echo "Environment variables set!"
jupyter notebook --no-browser
#jupyter notebook
