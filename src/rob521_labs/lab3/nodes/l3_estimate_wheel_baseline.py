#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
import time 

INT32_MAX = 2**31
NUM_ROTATIONS = 1
TICKS_PER_ROTATION = 4096
WHEEL_RADIUS = 0.066 / 2 #In meters


class wheelBaselineEstimator():
    def __init__(self):
        rospy.init_node('encoder_data', anonymous=True) # Initialize node

        #Subscriber bank
        rospy.Subscriber("cmd_vel", Twist, self.startStopCallback)
        rospy.Subscriber("sensor_state", SensorState, self.sensorCallback) #Subscribe to the sensor state msg

        #Publisher bank
        self.reset_pub = rospy.Publisher('reset', Empty, queue_size=1)

        #Initialize variables
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.del_left_encoder = 0
        self.del_right_encoder = 0
        self.isMoving = False #Moving or not moving
        self.lock = threading.Lock()

        #Reset the robot 
        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)

        # publish rotation
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1) # for calibration before the lab

        print('Ready to start wheel baseline calibration!')
        return

    def safeDelPhi(self, a, b):
        #Need to check if the encoder storage variable has overflowed
        diff = np.int64(b) - np.int64(a)
        if diff < -np.int64(INT32_MAX): #Overflowed
            delPhi = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
        elif diff > np.int64(INT32_MAX) - 1: #Underflowed
            delPhi = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
        else:
            delPhi = b - a  
        return delPhi

    def sensorCallback(self, msg):
        #Retrieve the encoder data form the sensor state msg
        self.lock.acquire()
        if self.left_encoder_prev is None or self.left_encoder_prev is None: 
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32
        else:
            #Calculate and integrate the change in encoder value
            self.del_left_encoder += self.safeDelPhi(self.left_encoder_prev, msg.left_encoder)
            self.del_right_encoder += self.safeDelPhi(self.right_encoder_prev, msg.right_encoder)

            #Store the new encoder values
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32
        self.lock.release()
        return

    def startStopCallback(self, msg):
        print(msg)
        if self.isMoving is False and np.absolute(msg.angular.z) > 0:
            self.isMoving = True  # Set state to moving
            print("Starting Calibration Procedure")

        elif self.isMoving is True and np.isclose(msg.angular.z, 0):
            self.isMoving = False  # Set the state to stopped
            # Calculate the separation of the wheels based on encoder measurements
            print('del_encoders', self.del_left_encoder, self.del_right_encoder)
            left_distance = (
                (self.del_left_encoder / TICKS_PER_ROTATION) * 2 * np.pi * WHEEL_RADIUS
            )
            right_distance = (
                (self.del_right_encoder / TICKS_PER_ROTATION) * 2 * np.pi * WHEEL_RADIUS
            )
            separation = (right_distance - left_distance) / (NUM_ROTATIONS * 2 * np.pi)
            print("Calibrated Separation: {} m".format(separation))

            # Reset the robot and calibration routine
            self.lock.acquire()
            self.left_encoder_prev = None
            self.right_encoder_prev = None
            self.del_left_encoder = 0
            self.del_right_encoder = 0
            self.lock.release()
            reset_msg = Empty()
            self.reset_pub.publish(reset_msg)
            print("Resetted the robot to calibrate again!")

        return

    def rotate_robot(self, angular_velocity):
        rate_hz = 10  # Frequency in Hz
        rate = rospy.Rate(rate_hz)
        twist_msg = Twist()
        twist_msg.angular.z = angular_velocity
        
        start_time = rospy.Time.now()  # Record start time

        while not rospy.is_shutdown():
            elapsed_time = (rospy.Time.now() - start_time).to_sec()
            if elapsed_time >= 6.28:
                break  # Stop after 6 seconds

            self.cmd_vel_pub.publish(twist_msg)
            rate.sleep()
        
        # Stop the robot after rotating for 6 seconds
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)


import threading

if __name__ == "__main__":
    Estimator = wheelBaselineEstimator()  # create instance

    # Define the angular velocity for rotation (radians per second)
    angular_velocity = 1  # Adjust this value as needed
    # rate_hz = 10
    # # Start rotation
    rotation_thread = threading.Thread(target=Estimator.rotate_robot, args=(angular_velocity,))

    # # Start the rotation thread
    rotation_thread.start()
        # Start rotation
    # Estimator.rotate_robot(angular_velocity)  # TODO: comment out for real deployment

    # # Assuming we want each rotation to take 2 seconds, wait for 6 seconds for 3 rotations
    # time.sleep(NUM_ROTATIONS * 2)

    # # Stop rotation
    # Estimator.rotate_robot(0)

    rospy.spin()
