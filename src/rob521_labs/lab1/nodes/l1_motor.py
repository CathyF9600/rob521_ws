#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    vel_twist = Twist()
    vel_twist.linear.x = 0.2
    vel_twist.angular.z = 0
    cur_distance = 0
    target_distance = 1
    rate = rospy.Rate(10)
    start_time = rospy.Time.now()
    rospy.loginfo(f'Starting time is {start_time}')
    while cur_distance < target_distance:
        vel_pub.publish(vel_twist)
        cur_time = rospy.Time.now()
        cur_distance = 0.2 * (cur_time - start_time).to_sec()
        rospy.loginfo(f'Current distance: {cur_distance:.2f} m')
        rate.sleep()
    vel_twist.linear.x = 0
    vel_pub.publish(vel_twist)

    # Rotation
    vel_twist.angular.z = -0.5
    cur_angle = 0
    target_angle = 6.28 # 360 deg
    start_time = rospy.Time.now()
    while cur_angle < target_angle:
        vel_pub.publish(vel_twist)
        cur_time = rospy.Time.now()
        cur_angle = 0.5 * (cur_time - start_time).to_sec()
        rospy.loginfo(f'Current angle: {cur_angle:.2f} rad')
        rate.sleep()
    vel_twist.angular.z = 0
    vel_pub.publish(vel_twist)
    rospy.loginfo('Robot stopped.')


    




def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
