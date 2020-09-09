rsync ros_overseer.py nao@pepper:~/ -rP
ssh nao@pepper
nohup python ros_overseer.py </dev/null >ros_overseer.log 2>&1 &
