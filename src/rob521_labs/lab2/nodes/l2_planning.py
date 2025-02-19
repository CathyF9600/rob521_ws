#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from scipy.spatial import cKDTree
import math

COLORS = dict(
    w=(255, 255, 255),
    k=(0, 0, 0),
    g=(0, 255, 0),
    r=(255, 0, 0),
    b=(0, 0, 255)
)

def load_map(filename):
    im = mpimg.imread("/home/rob521/catkin_ws/src/rob521_labs/lab2/maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("/home/rob521/catkin_ws/src/rob521_labs/lab2/maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 1 #m/s (Feel free to change!)
        self.rot_vel_max = 1 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        self.last_connected_id = 0
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return
    
    def distance_to_goal(self, cur_point):
        # print('cur_point', cur_point, self.goal_point)
        direction = self.goal_point.flatten()[:2] - cur_point[:2]
        distance = np.linalg.norm(direction)
        # print('direction', direction, direction.type)
        return np.array([distance])

    def distance_bw_points(self, p1, p2):
        x_g = p1[0] - p2[0]
        y_g = p1[1] - p2[1]
        return np.sqrt(x_g ** 2 + y_g ** 2)

    def distance_to_goal1(self, point):
        # goal distance
        x_g = self.goal_point[0] - point[0]
        y_g = self.goal_point[1] - point[1]

        return np.sqrt(x_g ** 2 + y_g ** 2)
    

    def sample_point_in_bounded_ellipse(self, start, end, x_bounds, y_bounds):
        """
        Sample a point inside an ellipse between `start` and `end`, ensuring it lies within given bounds.
        
        Parameters:
            start (np.ndarray): (2,1) array representing the start point [x; y].
            end (np.ndarray): (2,1) array representing the end point [x; y].
            x_bounds (tuple): (x_min, x_max) bounds.
            y_bounds (tuple): (y_min, y_max) bounds.

        Returns:
            np.ndarray: A sampled point (2,1) within the ellipse and bounds.
        """
        center = (start + end) / 2  # Center of the ellipse
        a = np.linalg.norm(end - start) / 2  # Semi-major axis (half of the distance)
        b = a / 2  # Arbitrary semi-minor axis (adjustable)

        while True:
            # Sample a random angle
            theta = np.random.uniform(0, 2 * np.pi)

            # Sample radius in [0,1] and scale it to fit within the ellipse
            r = np.random.uniform(0, a) ** 0.5  # Square root for uniform density
            x_offset = r * a * np.cos(theta)
            y_offset = r * b * np.sin(theta)

            # Compute the sampled point
            sampled_point = center + np.array([[x_offset], [y_offset]])

            # Extract x and y
            x, y = sampled_point[0, 0], sampled_point[1, 0]

            # Check if the sampled point is within the specified bounds
            if x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]:
                return sampled_point



    #Functions required for RRT
    def sample_map_space_bridge(self): # bridge
        #Return an [x,y] coordinate to drive the robot 
        # towards
        # print("TO DO: Sample point to drive towards")
        sample = np.zeros((2,1))
        tighter_bound = np.array([[0.0, 44], 
                                 [-46, 10]])
        
        dist_to_goal = self.distance_to_goal(self.nodes[self.last_connected_id].point) # the last node

        if dist_to_goal < 5:
            sample[0] = np.random.normal(self.goal_point[0], scale=5)  # Mean = goal, std dev = 2
            sample[1] = np.random.normal(self.goal_point[1], scale=5)
            c = COLORS['r']
        else:
            if len(self.nodes) > 4000: #dist_to_goal < 15: #15:
                
                # print('The sample is getting close to goal.')
                padding = dist_to_goal * 0.5
                # form a bound with range of [x_last - padding, goal + padding]
                x_lower_bound = max((self.goal_point[0, 0] - padding)[0], tighter_bound[0, 0])
                x_higher_bound = min((self.goal_point[0, 0] + padding)[0], tighter_bound[0, 1])
                # form a bound with range of [y_last - padding, goal + padding]
                y_lower_bound = max((self.goal_point[1, 0] - padding)[0], tighter_bound[1, 0])
                y_higher_bound = min( (self.goal_point[1, 0] + padding)[0], tighter_bound[1, 1])
                c = COLORS['r']

            else:
                
                padding = dist_to_goal * 0.4
                x_lower_bound = max((self.nodes[self.last_connected_id].point[0, 0] - padding)[0], tighter_bound[0, 0])
                x_higher_bound = min((self.nodes[self.last_connected_id].point[0, 0] + padding)[0], tighter_bound[0, 1])
                # form a bound with range of [y_last - padding, goal + padding]
                y_lower_bound = max((self.nodes[self.last_connected_id].point[1, 0] - padding)[0], tighter_bound[1, 0])
                y_higher_bound = min( (self.nodes[self.last_connected_id].point[1, 0] + padding)[0], tighter_bound[1, 1])
                    

            # # form a bound with range of [x_last - padding, goal + padding]
            # x_lower_bound = max((self.nodes[-1].point[0, 0] - padding)[0], tighter_bound[0, 0])
            # x_higher_bound = min((self.goal_point[0, 0] + padding)[0], tighter_bound[0, 1])
            # # form a bound with range of [y_last - padding, goal + padding]
            # y_lower_bound = max((self.goal_point[1, 0] - padding)[0], tighter_bound[1, 0])
            # y_higher_bound = min( (self.nodes[-1].point[1, 0] + padding)[0], tighter_bound[1, 1])

            sample[0] = np.random.uniform(low=x_lower_bound, high=x_higher_bound)
            sample[1] = np.random.uniform(low=y_lower_bound, high=y_higher_bound)
            sample_cand = []
            for _ in range(100):
                # print("x_lower_bound type:", type(x_lower_bound), x_lower_bound)
                # print("y_lower_bound type:", type(y_lower_bound), y_lower_bound)

                # assert np.isscalar(x_lower_bound) and np.isscalar(x_higher_bound), "Bounds must be scalars"
                # assert np.isscalar(y_lower_bound) and np.isscalar(y_higher_bound), "Bounds must be scalars"

                rand1 = sample
                rand2 = np.array([np.random.normal(sample[0], scale=1),  # Mean = goal, std dev = 2
                                np.random.normal(sample[1], scale=1)]).reshape(-1,1)
                # print('rand[0]', rand1, rand1.shape)

                if self.check_for_collision(rand1) and self.check_for_collision(rand2):  # Both points are free
                    midpoint = (rand1 + rand2) / 2
                    if not self.check_for_collision(midpoint):  # The midpoint is in an obstacle
                        sample = midpoint
                        sample_cand.append(sample)
                        # break  # Use this sample
                elif self.check_for_collision(rand1) or self.check_for_collision(rand2): 
                    if not self.check_for_collision(rand1):
                        sample = rand1
                    elif not self.check_for_collision(rand2):
                        sample = rand2
                    sample_cand.append(sample)
                    # break
            metric = self.distance_to_goal(sample)
            sample_final = sample
            print('len', len(sample_cand))
            for cand in sample_cand:
                # print('cand', cand[:2], self.goal_point)
                goal_metric = self.distance_to_goal(cand)
                if goal_metric < metric:
                    metric = goal_metric
                    sample_final = cand
            sample = sample_final
            c = COLORS['b']
        self.window.add_point(np.array(sample).reshape(2,), radius=2, width=0, color=c) # blue
        # input()
        return sample
    
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")
        sample = np.zeros((2,1))
        tighter_bound = np.array([[0.0, 44], 
                                 [-46, 10]])
        
        goal_bias_rate = 0.2
        cur_rand = np.random.rand()
        if cur_rand < goal_bias_rate:
            # Use goal-sampling occasionally
            sample = self.goal_point

        else:
            dist_to_goal = self.distance_to_goal(self.nodes[-1].point) # the last node
            if dist_to_goal < 5:
                sample[0] = np.random.normal(self.goal_point[0], scale=5)  # Mean = goal, std dev = 2
                sample[1] = np.random.normal(self.goal_point[1], scale=5)
            else:
                if dist_to_goal < 15:
                    # print('The sample is getting close to goal.')
                    padding = dist_to_goal * 1.2
                    # # form a bound with range of [x_last - padding, goal + padding]
                    # x_lower_bound = max((self.goal_point[0, 0] - padding)[0], tighter_bound[0, 0])
                    # x_higher_bound = min((self.goal_point[0, 0] + padding)[0], tighter_bound[0, 1])
                    # # form a bound with range of [y_last - padding, goal + padding]
                    # y_lower_bound = max((self.goal_point[0, 0] - padding)[0], tighter_bound[1, 0])
                    # y_higher_bound = min( (self.goal_point[0, 0] + padding)[0], tighter_bound[1, 1])
                    
                else:
                    padding = 0.2 * dist_to_goal
                    
                # form a bound with range of [x_last - padding, goal + padding]
                x_lower_bound = max((self.nodes[-1].point[0, 0] - padding)[0], tighter_bound[0, 0])
                x_higher_bound = min((self.goal_point[0, 0] + padding)[0], tighter_bound[0, 1])
                # form a bound with range of [y_last - padding, goal + padding]
                y_lower_bound = max((self.goal_point[1, 0] - padding)[0], tighter_bound[1, 0])
                y_higher_bound = min( (self.nodes[-1].point[1, 0] + padding)[0], tighter_bound[1, 1])

                sample[0] = np.random.uniform(low=x_lower_bound, high=x_higher_bound)
                sample[1] = np.random.uniform(low=y_lower_bound, high=y_higher_bound)
                # print("de", sample)
                # if cur_rand < bridge_rate:
                #     for _ in range(2):
                #         # print("x_lower_bound type:", type(x_lower_bound), x_lower_bound)
                #         # print("y_lower_bound type:", type(y_lower_bound), y_lower_bound)

                #         assert np.isscalar(x_lower_bound) and np.isscalar(x_higher_bound), "Bounds must be scalars"
                #         assert np.isscalar(y_lower_bound) and np.isscalar(y_higher_bound), "Bounds must be scalars"

                #         rand1 = sample
                #         rand2 = np.array([np.random.normal(sample[0], scale=2),  # Mean = goal, std dev = 2
                #                         np.random.normal(sample[1], scale=2)]).reshape(-1,1)
                #         # print('rand[0]', rand1, rand1.shape)

                #         if self.check_for_collision(rand1) and self.check_for_collision(rand2):  # Both points are free
                #             midpoint = (rand1 + rand2) / 2
                #             if not self.check_for_collision(midpoint):  # The midpoint is in an obstacle
                #                 sample = midpoint
                #                 break  # Use this sample

        self.window.add_point(np.array(sample).reshape(2,), radius=1, width=0, color=(0, 0, 255)) # blue
        return sample
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")
        for node in self.nodes:
            if np.linalg.norm(node.point[:2] - point[:2]) < 0.5:
                return True
        return False

    def closest_node(self, point):
        # print("TO DO: Implement a method to get the closest node to a sapled point")
        # point is [x, y]
        #Returns the index of the closest node
        tree_nodes = np.array([node.point[:2] for node in self.nodes]).reshape(-1, 2)
        # print('tree', tree_nodes)
        # input()
        tree = cKDTree(tree_nodes) # takes N x 2
        _, i = tree.query(point.T, k=1)
        return i[0]
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        vel, rot_vel = self.robot_controller(node_i, point_s)
        # print('vel', self.robot_controller(node_i, point_s), self.robot_controller1(node_i, point_s))
        # input()
        # v, r = self.master_controller(node_i, vel, rot_vel)
        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        K_linear = 0.5
        K_heading = 0.25
        K_angular = 0.5
        direction = (point_s - node_i[:2]).flatten()
        distance = np.linalg.norm(direction)
        theta = np.arctan2(direction[1], direction[0])
        heading_e = theta - node_i[2, 0]
        # print('distance', direction, heading_e)
        # print('heading_e', heading_e)

        if heading_e > math.pi: # keep it bw  -pi and pi
            heading_e = heading_e - 2 * math.pi
        elif heading_e < -math.pi:
            heading_e = heading_e + 2 * math.pi

        # input()
        vel = min(K_linear * distance - K_heading * abs(heading_e), self.vel_max)
        if heading_e > 0:
            rot_vel = min(K_angular * heading_e, self.rot_vel_max)
        else:
            rot_vel = max(K_angular * heading_e, -self.rot_vel_max)
        return vel, rot_vel

    def master_controller(self, point_s, vel, rot_vel):
        # with the vel rot vel pair
        substep_time = self.timestep / self.num_substeps
        vel_l, rotv_l = [np.array([vel])], [np.array([rot_vel])]
        print('point_s', point_s)
        heading_e = point_s[2, 0] + rot_vel
        for i in range(self.num_substeps-1):
            x_velos = vel * np.cos(heading_e)
            y_velos = vel * np.sin(heading_e)
            # print('contorl ', x_velos, y_velos)
            x_substep_displacements = substep_time * x_velos
            y_substep_displacements = substep_time * y_velos
            cur_p = np.array([x_substep_displacements, y_substep_displacements, heading_e]).reshape(-1, 1)
            vel, rot_vel = self.robot_controller(cur_p, point_s[:2])
            # find heading error
            direction = (point_s[:2] - cur_p[:2]).flatten()
            theta = np.arctan2(direction[1], direction[0])
            heading_e = theta - cur_p[2, 0]
            vel_l.append(np.array([vel]))
            rotv_l.append(np.array([rot_vel]))
        v, r = np.array(vel_l), np.array(rotv_l)
        print('v', v)
        input('master')
        return v, r

    
    def trajectory_rollout(self, vel, rot_vel, cur_pos):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        substep_time = self.timestep / self.num_substeps
        # print("vel", vel,rot_vel) # 0.5 -0.2
        init_heading = cur_pos[2, 0] # get theta from 3 by 1 vector [x;y;theta]
        final_heading = init_heading + rot_vel
        init_x = cur_pos[0, 0]
        init_y = cur_pos[1, 0]
        headings = np.linspace(init_heading, final_heading, num=self.num_substeps).reshape(-1, 1)
        # Find instantaneous linear velocities at each trajectory subpoint
        x_velos = np.zeros((self.num_substeps, 1)) # first step 0 linear velocity (coresponding to initial heading)
        # print('headings[0:-1]', headings[0:-1].shape)
        x_velos[1:] = vel * np.cos(headings[0:-1]) # slice all except the last heading
        # print('vel * np.cos(headings[0:-1])', vel , np.cos(headings[0:-1]), vel * np.cos(headings[0:-1]), self.num_substeps)
        y_velos = np.zeros((self.num_substeps, 1))
        y_velos[1:] = vel * np.sin(headings[0:-1])
        
        # Multiply velocities by substep duration for distance travelled between substeps
        x_substep_displacements = substep_time * x_velos
        y_substep_displacements = substep_time * y_velos
        
        # cumsum and add initial position to get trajectory subpoint locations
        x_points = np.reshape(init_x + np.cumsum(x_substep_displacements), (self.num_substeps, 1))
        y_points = np.reshape(init_y + np.cumsum(y_substep_displacements), (self.num_substeps, 1))
                        
        # Stack
        trajectory = np.hstack((x_points, y_points, headings))
        # print('traj', trajectory, trajectory.shape) # (10, 3)
        return trajectory.T # 3 x 10

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        origin = np.array(self.map_settings_dict["origin"][:2]).reshape(2, 1)
        scale = self.map_settings_dict["resolution"]

        indices = (point - origin) / scale
        indices[1, :] = self.map_shape[0] - indices[1, :] # world frame to grid frame
        # print('np.floor(indices).astype(int)', origin, np.floor(indices).astype(int))
        return np.floor(indices).astype(int)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")
        robot_indices_grid = self.point_to_cell(points)
        robot_r_grid = self.robot_radius / self.map_settings_dict["resolution"]
        rr, cc = [], []

        for i in range(0, len(robot_indices_grid[0])):
            center = (robot_indices_grid[0][i], robot_indices_grid[1][i])
            # disk returns the row (rr_o) and column (cc_o) indices of all pixels inside a circular region centered at center with radius robot_r
            rr_o, cc_o = disk(center, robot_r_grid)
            # print('rr', rr_o, cc_o)
            rr_o = np.clip(rr_o, 0, self.map_shape[1] - 1) #limits values to be between 0 and map size -1
            cc_o = np.clip(cc_o, 0, self.map_shape[0] - 1)
            rr = np.concatenate((rr, rr_o)).astype(int)
            cc = np.concatenate((cc, cc_o)).astype(int)

        return rr, cc

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        # print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        trajectory = self.simulate_trajectory(node_i, point_f)
        print('trajectory', trajectory)
        return trajectory
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        # print("TO DO: Implement a cost to come metric")
        return np.sum(np.linalg.norm(np.diff(trajectory_o[:2], axis=1), axis=0)) # Euclidean Distance
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")
        for child_id in self.nodes[node_id].children_ids:
            new_cost = self.nodes[node_id].cost + np.linalg.norm(self.nodes[child_id].point[:2] - self.nodes[node_id].point[:2])
            if new_cost < self.nodes[child_id].cost:
                self.nodes[child_id].cost = new_cost
                self.update_children(child_id)


    def update_children1(self, node_id, old_cost):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")
        node = self.nodes[node_id]

        for child_id in node.children_ids:
            child = self.nodes[child_id]
            old_child_cost = child.cost
            node_to_child_cost = old_child_cost - old_cost
            child.cost = node.cost + node_to_child_cost
            
            # Recursively update grandchildren
            self.update_children1(child_id, old_child_cost)

        return
    


    # Functions required for RRT
    def check_for_collision(self, trajectory): # true if collide
        #Given a trajectory - a list of points
        #Return boolean if this trajectory is collision free
        
        all_rr, all_cc = self.points_to_robot_circle(trajectory[:2, :])
        map_pixels = self.occupancy_map[all_cc[:], all_rr[:]]
        
        if np.max(all_rr) > self.map_shape[1] or np.max(all_cc) > self.map_shape[0] or np.min(all_rr) < 0 or np.min(all_cc) < 0:
            return True # robot collides with bounds
            
        if np.min(map_pixels) == 1: # all white space (Whitespace is true, black is false)
            '''
            for i in range(trajectory.shape[1]):
                self.window.add_point(trajectory[:2, i], radius=4, width=0, color=COLORS['g'])
            '''    
            return False # no collision
        else:
  
            return True # collision detected


    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            print(i)
            #Sample map space
            point = self.sample_map_space()
            #print(i, point)
            curr_id = len(self.nodes) # the node id will be the array index

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            collision = self.check_for_collision(trajectory_o)
            duplicate = self.check_if_duplicate(trajectory_o[:, -1])

            if not collision and not duplicate:
                self.nodes.append(Node(trajectory_o[:, -1].reshape(3,-1), closest_node_id, cost=0))
                self.nodes[closest_node_id].children_ids.append(curr_id)
                self.last_connected_id = curr_id
                # visualizing the valid trajectory
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(np.array(trajectory_o[:, -1].reshape((3,))))
                for pt in temp_pt:
                    self.window.add_point(np.copy(pt), color=COLORS['g'])
                #Check if goal has been reached
                print('distance', self.distance_to_goal(trajectory_o[:, -1]))
                if self.distance_to_goal(trajectory_o[:, -1]) < self.stopping_dist:
                    print("RRT FINISHED")
                    path = self.recover_path()

                    return path
        return []
    

    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot   
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            print(i)
            #Sample
            point = self.sample_map_space()
            curr_id = len(self.nodes) # the node id will be the array index

            #Closest Node
            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(closest_node.point, point)

            #Check for Collision and duplicate
            newest_node = None
            if not self.check_for_collision(trajectory_o) and (not self.check_if_duplicate(trajectory_o[:, -1])):
                # wire the new sample to the closest node
                cost = self.cost_to_come(trajectory_o) + closest_node.cost
                newest_node = Node(trajectory_o[:,-1].reshape(3,1), closest_node_id, cost)#, traj=trajectory_o[0:2,:])
                self.nodes.append(newest_node)
                closest_node.children_ids.append(curr_id)
                #self.trajectories.append(trajectory_o)
                
                # visualizing the valid trajectory
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(np.array(trajectory_o[:, -1].reshape((3,))))
                for pt in temp_pt:
                    self.window.add_point(np.copy(pt), color=COLORS['g'])
                
            if(newest_node != None):
                # Get candidate nodes near the newest node for rewiring
                # Note: Exclude first and last nodes from this check
                ball_r = self.ball_radius()
                candidates = []
                for i in range(1, len(self.nodes)-2):
                    dist = np.linalg.norm(self.nodes[i].point[0:2]-newest_node.point[0:2])
                    if(dist <= ball_r and i != newest_node.parent_id):
                        candidates.append(i)
                
                # Perform rewiring on candidates if it improves cost-to-come
                for cand in candidates:
                    candidate_node = self.nodes[cand]
                    traj = self.connect_node_to_point(newest_node.point, candidate_node.point[0:2])
                    if not np.array_equal(traj, np.zeros((3, self.num_substeps))):
                        # Calculate costs
                        traj_cost = self.cost_to_come(traj)
                        new_cost = newest_node.cost + traj_cost
                        old_cost = candidate_node.cost
                        
                        if(new_cost < old_cost):
                            candidate_node.cost = new_cost
                            
                            # Update parent-child relationships
                            self.nodes[candidate_node.parent_id].children_ids.remove(cand)
                            newest_node.children_ids.append(cand)
                            candidate_node.parent_id = len(self.nodes)-1
                            
                            # # visualizing the rewiring
                            # #for pt in candidate_node.parent_trajectory.T:
                            # #    self.window.add_point(np.copy(pt), color=COLORS['w'])
                            # #candidate_node.parent_trajectory = traj[0:2,:]                       
                            # temp_pt = np.array(traj[0:2, :]).copy().T
                            # self.window.add_se2_pose(np.array(traj[:, -1].reshape((3,))))
                            # for pt in temp_pt:
                            #     self.window.add_point(np.copy(pt), color=COLORS['r'])
                        
                            self.update_children1(cand, old_cost)
                            # #print("FINISHED UPDATING")
            
                #Check for early end
                if self.distance_to_goal(trajectory_o[:, -1]) < self.stopping_dist:
                    print("RRT* FINISHED")
                    path = self.recover_path()
                    return path
        return self.nodes
    

    def rrt_star_planning1(self):
        #This function performs RRT* for the given map and robot        
        from queue import PriorityQueue

        pq = PriorityQueue()

    
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            print(len(self.nodes))
            #Sample
            # point = self.sample_map_space_bri()
            point = self.sample_map_space()
            curr_id = len(self.nodes)
            #Closest Node
            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]
            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            cost_with_closest = self.cost_to_come(trajectory_o) + closest_node.cost
            #Check for Collision
            collision = self.check_for_collision(trajectory_o)
            duplicate = self.check_if_duplicate(trajectory_o[:, -1])

            if not collision and not duplicate:
                ball_r = self.ball_radius()
                neighbors = [] # ids
                for i in range(1, len(self.nodes)-1):
                    dist = np.linalg.norm(self.nodes[i].point[:2] - point)
                    # traj_n = self.simulate_trajectory(self.nodes[i].point, point)
                    # cost_with_neighbor = self.cost_to_come(traj_n)
                    # print('ball_r', ball_r, dist)
                    # input()
                    if dist <= ball_r and i != closest_node_id:
                        neighbors.append(i)
                #Last node rewire
                #Default is the closest node
                best_neighbor_id = closest_node_id
                lowest_cost_so_far = cost_with_closest
                for neighbor_id in neighbors:
                    traj_n = self.simulate_trajectory(self.nodes[neighbor_id].point, point)
                    collision_neighbor = self.check_for_collision(traj_n)
                    if not collision_neighbor:
                        cost_with_neighbor = self.cost_to_come(traj_n) + self.nodes[neighbor_id].cost
                        if cost_with_neighbor < lowest_cost_so_far:
                            best_neighbor_id = neighbor_id
                            lowest_cost_so_far = cost_with_neighbor
                # Wire with the neighbor with lowest cost to come
                self.nodes.append(Node(trajectory_o[:, -1].reshape(3,-1), best_neighbor_id, cost=lowest_cost_so_far))
                self.nodes[best_neighbor_id].children_ids.append(curr_id)
                self.nodes[-1].parent_id = best_neighbor_id

                # visualizing the valid trajectory
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(np.array(trajectory_o[:, -1].reshape((3,))))
                for pt in temp_pt:
                    self.window.add_point(np.copy(pt), color=COLORS['g'])

                #Close node rewire
                for neighbor_id in neighbors:
                    neighbor_node = self.nodes[neighbor_id]
                    cur_cost = neighbor_node.cost
                    traj_c = self.simulate_trajectory(neighbor_node.point, point)
                    if not self.check_for_collision(traj_c):
                        new_cost = self.nodes[-1].cost + self.cost_to_come(traj_c)
                        if new_cost < cur_cost:
                            # rewire
                            self.nodes[neighbor_node.parent_id].children_ids.remove(neighbor_id)
                            self.nodes[-1].children_ids.append(neighbor_id)
                            neighbor_node.parent_id = curr_id

                            # # visualizing the rewiring
                            # #for pt in candidate_node.parent_trajectory.T:
                            # #    self.window.add_point(np.copy(pt), color=COLORS['w'])
                            # #candidate_node.parent_trajectory = traj[0:2,:]                       
                            # temp_pt = np.array(traj_c[0:2, :]).copy().T
                            # self.window.add_se2_pose(np.array(traj_c[:, -1].reshape((3,))))
                            # for pt in temp_pt:
                            #     self.window.add_point(np.copy(pt), color=COLORS['r'])
                        
                            # self.update_children(neighbor_id)
                            self.update_children1(neighbor_id, cur_cost)

                #Check for early end
                print('distance', self.distance_to_goal(trajectory_o[:, -1]))
                if self.distance_to_goal(trajectory_o[:, -1]) < self.stopping_dist:
                    print("RRT* FINISHED")
                    path = self.recover_path()
                    pq.put((self.cost_to_come(trajectory_o), [path, trajectory_o]))
                    # return path
        print('pq', pq)
        _, res = pq.get()
        return res[0]
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path
    
    # def recover_path(self, node_id=-1):
    #     path = []
    #     while node_id != -1:
    #         path.append(self.nodes[node_id].point.reshape(3, 1))  # Ensure shape is (3,1)
    #         node_id = self.nodes[node_id].parent_id
    #     return np.hstack(path) if path else np.empty((3, 0))  # Ensure return is a (3, N) array


def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42], [-44]]) #m # np.array([[10], [10]])
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    path = path_planner.rrt_planning()
    # path = path_planner.rrt_star_planning1()

    np.savetxt("mapfile.txt", path_planner.occupancy_map)
    print('path', path)
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)

    #Leftover test functions
    if len(path) != 0:
        print("Success")
        path = np.array(path)
        print(path.shape)
        # np.save("willowgarage_rrt.npy", path)
        np.save("willowgarage_rrt_star.npy", path)

        for i in range(path.shape[0]):
            # print('path[:2, i]', path[i, :2])
            path_planner.window.add_point(np.copy(path[i, :2]).reshape(2,), radius=4, width=0, color=COLORS['r'])

    # pygame.image.save(path_planner.window.screen, "willowgarageworld_rrt.png")
    pygame.image.save(path_planner.window.screen, "willowgarageworld_rrt_star.png")


if __name__ == '__main__':
    main()
