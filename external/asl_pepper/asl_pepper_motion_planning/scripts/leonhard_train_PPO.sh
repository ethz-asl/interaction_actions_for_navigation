# bsub -W 120:00 -R "rusage[ngpus_excl_p=1, mem=8192]" < 4.sh
export MACHINE_NAME="leonhard"
cd ~
module load eth_proxy python_gpu/3.6.4
pip install --user -e ~/Documents/baselines
pip install --user numba
# install range_libc
pip install --user cython
cd ~/Documents/range_libc/pywrapper
WITH_CUDA=ON python setup.py install --user
python setupcmap2d.py install --user
# pip install --user pycuda
cd ~/Documents/pepper_ws/src/asl_pepper/asl_pepper_motion_planning/scripts
for i in {1..100}
do
  python trainPPO.py --mode BOUNCE --map-name office_full --reward-collision -5
  if [[ $? -eq 139 ]]; then
    echo "oops, sigsegv";
  else
    break
  fi
done
