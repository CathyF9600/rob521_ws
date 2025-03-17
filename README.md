# rob521
## lab2
- `roslaunch rob521_lab2 map_view.launch`
- `cd ~/catkin_ws && catkin_make && source devel/setup.bash && cd src/rob521_labs/lab2/nodes/ && rosrun rob521_lab2 l2_follow_path_willow.py`

## lab3
1. Calibration
 - roscore
 - rosrun rob521_lab3 l3_estimate_wheel_baseline.py
 - rosrun rob521_lab3 l3_estimate_wheel_radius.py
 - rosbag play sample_data.bag --pause
task 5
    - put the robot on the ground with a 1m circle space
    - on real robot: `roslaunch turtlebot3_bringup turtlebot3_robot.launch --screen`
    - on remote PC: `rosrun rob521_lab3 l3_estimate_robot_motion.py`
    - check terminal to make sure ur and turtleBot's estimates are close to zero
    - Within 5 seconds, on remote PC: `roslaunch rob521_lab3 wheel_odom_rviz.launch` 
    - if you want to teleoperate instead, `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch`
    - once you close the estimator node, to play the rosbag `rosrun rob521_lab3 l3_plot_motion_estimate.py` contains both mine and onboard pose estimates

task 6
    - on robot: `roslaunch turtlebot3_bringup turtlebot3_robot.launch --screen`
    - on remote PC: `roslaunch rob521_lab3 mapping_rviz.launch`
    - `rosrun rob521_lab3 l3_mapping.py`
    - to teleoperate in myhal `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch`