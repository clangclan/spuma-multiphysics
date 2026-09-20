source /home/jsw/cae-gpu/spuma-env.sh || return $?
source /home/jsw/cae-gpu-pr1-compatible/spuma/etc/bashrc "" || return $?
export FOAM_EXTRA_CFLAGS=-tp=x86-64-v3
export FOAM_EXTRA_CXXFLAGS=-tp=x86-64-v3
export have_cuda=true NVARCH=120
export OMP_NUM_THREADS=1
export CANTERA_DATA=/home/jsw/cae-gpu-pr1-compatible/cantera-data
export FOAM_SIGFPE=false
