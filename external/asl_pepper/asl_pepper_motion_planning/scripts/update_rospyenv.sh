# Python dependencies
source ~/peppervenv/bin/activate
cd ~/Code/pepper_ws/src/asl_pepper/asl_pepper_2d_simulator/python
pip install -e .
cd ~/Code/pepper_ws/src/pyniel || cd ~/Code/pyniel
git pull --ff-only
pip install .
cd ~/Code/pepper_ws/src/range_libc/pywrapper
git pull --ff-only
python setup.py install
cd ~/Code/pepper_ws/src/pymap2d
git pull --ff-only
pip install .
cd ~/Code/pepper_ws/src/pylidar2d
git pull --ff-only
pip install .
cd ~/Code/pepper_ws/src/interaction_actions/python/cIA
git pull --ff-only
pip install .
cd ..
pip install -e .
cd ~/Code/pepper_ws/src/responsive/lib_dwa
git pull --ff-only
pip install .
cd ../lib_clustering
pip install .
