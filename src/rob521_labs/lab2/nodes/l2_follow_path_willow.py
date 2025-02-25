#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils
import math
from skimage.draw import disk
import random

TRANS_GOAL_TOL = .2  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .5  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
VEL_MAX = 0.26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
ROT_VEL_MAX = 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.15  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'shortest_path_rrt_good.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
# TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


'''
given current pose (node_i, [x;y;theta]) and a goal point (point_s, [x;y])
determine a lin and rotational velocity to drive the robot towards the goal point over timestep self.timestep
without exceeding max velocities (self.vel_max, self.rot_vel_max)
'''
def robot_controller(node_i, point_s):
    #This controller determines the velocities that will nominally move the robot from node i to node s
    #Max velocities should be enforced
    
    # P controller gains. TUNE
    # K_linear = 0.2
    # K_heading = 0.2
    # K_angular = 0.2

    K_linear = 0.5
    K_angular = 0.5
    K_heading = 0.5
    
    # get distance between node_i and point_s
    distance = np.linalg.norm(np.array([[point_s[0]-node_i[0]],
                                    [point_s[1]-node_i[1]]]))
    
    # get angle between x axis and vector from node_i to point_s
    desired_travel_dir = np.arctan2(point_s[1]-node_i[1], point_s[0]-node_i[0])
    heading_e = desired_travel_dir - node_i[2]
    
    print("Heading Error and Desired Travel Dir")
    print(heading_e)
    print(desired_travel_dir)
    
    if heading_e > math.pi: # keep it bw  -pi and pi
        heading_e = heading_e - 2 * math.pi
    elif heading_e < -math.pi:
        heading_e = heading_e + 2 * math.pi

    # input()
    # vel = min(K_linear * distance - K_heading * abs(heading_e), VEL_MAX)
    vel = max(min(VEL_MAX, K_linear * distance - K_heading * abs(heading_e)), 0.05) # assume it doesnt go backwards
    # # avoid close to 0 vel
    # if vel < 0:
    #     vel = min(vel, -0.05)
    # elif vel > 0:
    #     vel = max(vel, 0.05)
    # else:
    #     vel = 0.05
    if heading_e > 0:
        rot_vel = min(K_angular * heading_e, ROT_VEL_MAX)
    else:
        rot_vel = max(K_angular * heading_e, -ROT_VEL_MAX)
    
    #print("TO DO: Implement a control scheme to drive you towards the sampled point")
    return vel, rot_vel

#Map Handling Functions
def load_map(filename):
    import matplotlib.image as mpimg
    import cv2 
    im = cv2.imread("../maps/" + filename)
    im = cv2.flip(im, 0)
    # im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    im_np = np.logical_not(im_np)     #for ros
    return im_np

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        # map = rospy.wait_for_message('/map', OccupancyGrid)
        # self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        # self.map_resolution = round(map.info.resolution, 5)
        # self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        # self.map_nonzero_idxes = np.argwhere(self.map_np)
        map_filename = "willowgarageworld_05res.png"
        occupancy_map = load_map(map_filename)
        self.map_np = occupancy_map
        self.map_resolution = 0.05
        self.map_origin = np.array([ 0.2 , 0.2 ,-0. ])
        self.map_nonzero_idxes = np.argwhere(self.map_np)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    '''
    Assume points is a 2 by N matrix of points of interest
    '''
    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")
        
        scaled_radius = self.collision_radius_pix
        all_col_inds = np.array([1])
        all_row_inds = np.array([1])
        
        for i in range(0, np.shape(points)[0]):
            center = (points[i][0], points[i][1])
            row_inds, col_inds = disk(center, scaled_radius)
            all_col_inds = np.concatenate((all_col_inds, col_inds), axis=0)
            all_row_inds = np.concatenate((all_row_inds, row_inds), axis=0)
        
        all_col_inds = np.delete(all_col_inds, 0, 0)
        all_row_inds = np.delete(all_row_inds, 0, 0)
        
        return all_row_inds, all_col_inds
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            
            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)   # Initializes the first node of each trajectory to be the curr pose
            '''
            print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            for t in range(1, self.horizon_timesteps + 1):
                # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
                pass
            '''
            
            # self.horizon_timesteps = number of substeps of trajectory. 
            # self.num_opts -> number of trajectories to review
            # self.pose_in_map_np -> current robot pose
            # self.cur_goal -> target of this trajectory rollout
            cur_position = self.pose_in_map_np.reshape((3,1))
            heur_lin_vel, heur_ang_vel = robot_controller(cur_position, np.array([[self.cur_goal[0]],[self.cur_goal[1]]]))
            
            # Use the raw heuristic velocities for the first trajectory. 
            # Then randomly sample velocities within a range centered about these heuristic values for other trajectories
            lin_vel = heur_lin_vel
            ang_vel = heur_ang_vel
            substep_duration = INTEGRATION_DT
            
            # Get a set of lin and ang velocities sampled from a range centered about the heuristic velocities
            LIN_VEL_RNG = 0.05
            ANG_VEL_RNG = 0.1
            vel_opt_pairs = []
            vel_opt_pairs.append([lin_vel, ang_vel])
            
            # For each trajectory
            for i in range(0, self.num_opts):
                
                '''
                start of trajectory rollout
                TO DO: Vectorize for loops (the non-edge case one)
                    ALT: If the straight then turn matrix multiplication turns out to be a good enough approx, vectorize the whole thing
                '''
                
                # Get robot heading at each trajectory subpoint
                init_heading = self.pose_in_map_np[2]
                final_heading = init_heading + ang_vel * CONTROL_HORIZON
                headings = np.linspace(init_heading, final_heading, num=self.horizon_timesteps+1).reshape(-1,1)
                
                # Find instantaneous linear velocities at each trajectory subpoint
                x_vel = np.zeros((self.horizon_timesteps+1, 1))
                x_vel[1:] = lin_vel * np.cos(headings[0:-1])
                y_vel = np.zeros((self.horizon_timesteps+1, 1))
                y_vel[1:] = lin_vel * np.sin(headings[0:-1])
                
                # Multiply velocities by substep duration for distance travelled between substeps
                x_substep_displacements = substep_duration * x_vel
                y_substep_displacements = substep_duration * y_vel
                
                # cumsum and add initial position to get trajectory subpoint locations
                x_points = np.reshape(self.pose_in_map_np[0] + np.cumsum(x_substep_displacements), (self.horizon_timesteps+1, 1))
                y_points = np.reshape(self.pose_in_map_np[1] + np.cumsum(y_substep_displacements), (self.horizon_timesteps+1, 1))
                                
                # Stack
                local_paths[:,i,:] = np.hstack((x_points, y_points, headings))
                
                lin_vel = random.uniform(heur_lin_vel-LIN_VEL_RNG, heur_lin_vel+LIN_VEL_RNG)
                ang_vel = random.uniform(heur_ang_vel-ANG_VEL_RNG, heur_ang_vel+ANG_VEL_RNG)
                if(lin_vel > VEL_MAX):
                    lin_vel = VEL_MAX
                if(lin_vel < 0.05):
                    lin_vel = 0.05
                if(ang_vel > ROT_VEL_MAX):
                    ang_vel = ROT_VEL_MAX
                if(ang_vel < -ROT_VEL_MAX):
                    ang_vel = -ROT_VEL_MAX
                vel_opt_pairs.append([lin_vel, ang_vel])
                    
            # Skipping collision checking until end for efficiency (lazy collision checking)
            # calculate final cost and choose best option
            #print("TO DO: Calculate the final cost and choose the best control option!")
            
            final_costs = []
            
            for opt_idx in range(local_paths.shape[1]):
                final_costs.append(np.linalg.norm(self.cur_goal[0:2] - local_paths[-1,opt_idx,0:2]))
                
            min_cost = min(final_costs)
            best_opt_idx = final_costs.index(min_cost)
            
            # Collision check for the best path. If it fails, take the next best path and repeat until chosen path passes
            final_costs_copy = final_costs.copy()
            final_costs_copy.pop(best_opt_idx)
            collision = True
            
            while(collision and len(final_costs_copy) > 0):
                local_paths_pixels = (self.map_origin[:2] + local_paths[:, best_opt_idx, :2]) / self.map_resolution
                row_inds, col_inds = self.points_to_robot_circle(local_paths_pixels)
                trajectory_cells = self.map_np[col_inds, row_inds]
                # print('debug', np.max(row_inds) > np.shape(self.map_np)[1], np.max(col_inds) > np.shape(self.map_np)[0])
                # print(np.min(row_inds) < 0,  np.min(col_inds) < 0)

                if np.max(row_inds) > np.shape(self.map_np)[1] or np.max(col_inds) > np.shape(self.map_np)[0]: # or np.min(row_inds) < 0 or np.min(col_inds) < 0:
                    collision = True # Out of bounds
                    print('Out of bounds')
                    min_cost = min(final_costs_copy)
                    final_costs_copy.remove(min_cost)
                    best_opt_idx = final_costs.index(min_cost)
                else:
                    collision = False    
                    # if(np.max(trajectory_cells) == 0):
                    #     collision = False                    
                    # else: 
                    #     print('collision is still True')
                    #     # input()
                    #     min_cost = min(final_costs_copy)
                    #     final_costs_copy.remove(min_cost)
                    #     best_opt_idx = final_costs.index(min_cost)

            if collision:  # if all options fail, back off
                control = [-0.1, 0]
            else:
                control = vel_opt_pairs[best_opt_idx]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt_idx], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))
            # input()

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass