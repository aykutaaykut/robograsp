# Robotic Grasping with Deep Q-Learning

This is our Computer Engineering Senior Design Project at Koç University. Robotic grasping is a sequence of complicated actions and observations of the real world. Our designed and trained Deep Q-Learning agent had become to overcome this problem with 93% accuracy in simple shaped object trials and achieved 50 % accuracy in complex shaped object trials. Poster and final report of the project with all details are documented.

In order to run the project, you need to run the following commands in different terminal windows:
- roslaunch panda_sim_learning world.launch
- rosrun pc_segmentation def_loop -rt /camera/points -v 0 -dt 0.5 -ct 5 -t 5 -e 0.1
- gzclient
