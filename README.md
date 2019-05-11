roslaunch panda_sim_learning world.launch

rosrun pc_segmentation def_loop -rt /camera/points -v 0 -dt 0.5 -ct 5 -t 5 -e 0.1

gzclient
