#!/usr/bin/env python3
#

from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
import math
import numpy
import cv2
import heapq

class Turtlebot3AStar(Node):

    def __init__(self):
        super().__init__('turtlebot3_a_star_node')
        
        self.a_star_map = UInt8MultiArray()
        self.map2D = UInt8MultiArray()
        self.start_goal_coords = UInt8MultiArray()
        self.threshold = 75

        self.map_recieved = False
        self.coords_recieved = False

        self.coords = []

        qos = QoSProfile(depth=10)

        self.path_list_pub = self.create_publisher(UInt8MultiArray, 'path_list', qos) #TODO

        self.start_goal_sub = self.create_subscription(
            UInt8MultiArray,
            'start_goal_coords',
            self.start_goal_coords_callback,
            qos_profile=qos_profile_sensor_data)

        self.start_goal_sub = self.create_subscription(
            UInt8MultiArray,
            'a_star_map',
            self.a_star_map_callback,
            qos_profile=qos_profile_sensor_data)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        
    def a_star_map_callback(self, msg):
        self.a_star_map = msg
        self.map_recieved = True

    def start_goal_coords_callback(self, msg):
        self.start_goal_coords = msg
        self.coords_recieved = True

    
    def timer_callback(self):
        if self.map_recieved and self.coords_recieved:
            self.map_recieved = False
            self.coords_recieved = False
            self.map2D = self.multi_array_deconstructor(self.a_star_map)
            self.coords = self.multi_array_deconstructor(self.start_goal_coords)
            coords = self.multi_array_deconstructor(self.start_goal_coords)
            self.a_star((coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]))
            # self.a_star_chat((coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]))

    def multi_array_deconstructor(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size

        # Reconstruct the 2D array from flat data list
        data = msg.data
        array_2d = [data[i * cols:(i + 1) * cols] for i in range(rows)]
        
        return array_2d

    # construct multiarray message
    def multi_array_constructor(self, array):
        msg = UInt8MultiArray()

        # Flatten the 2D array for the 'data' field
        msg.data = [int(item) for sublist in array for item in sublist]

        # Define the layout dimensions
        dim_row = MultiArrayDimension()
        dim_row.label = "rows"
        dim_row.size = len(array)
        dim_row.stride = len(array) * len(array[0])

        dim_col = MultiArrayDimension()
        dim_col.label = "cols"
        dim_col.size = len(array[0])
        dim_col.stride = len(array[0])

        msg.layout.dim = [dim_row, dim_col]
        msg.layout.data_offset = 0

        self.get_logger().info('Constructed multi-dimensional array')
        return msg
    
    
    # Tar in cameFrom listan och den nuvarande koordinaten som borde vara mål-koordinaten och
    # bygger tillbaka vägen från mål till start genom att följa tidigare koordinater.
    def reconstruct_path(self, cameFrom, current, totalPath):
        totalPath.append(current)

        # Bygger tillbaka vägen från målet till start
        while True:
            for coord, prev_coord in cameFrom:
                if coord == current:
                    previous = prev_coord
                    break
                else:
                    previous = None
            
            if previous == None:
                break
            
            current = previous
            totalPath.insert(0, current)
        
        print('Yay!')

    # Heuristikfunktionen som tar in två koordinater, a och b, och beräknar avståndet mellan de.
    def heuristic(self, a, b):
        # Manhattan distance as self.heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Tar in en av scorlistorna, gScore eller fScore, en koordinat och ett nytt värde som den
    # koordinaten ska få och uppdaterar eller lägger till den i listan.
    def set_score(self, scoreList, coord, newScore):
        if scoreList == []:
            scoreList.append((coord, newScore))
            return

        for i in range(len(scoreList)):
            if scoreList[i][0] == coord:
                scoreList[i] = (coord, newScore)
                return

        scoreList.append((coord, newScore))

    # Tar in en av scorlistorna, gScore eller fScore och en koordinat och returnerar värdet som den
    # koordinaten har i listan. Om den inte finns returneras oändligheten.
    def get_score(self, scoreList, coord):
        score = math.inf

        for n, s in scoreList:
            if n == coord:
                score = s

        return score

    # Tar in openSet listan och fScore listan och returnerar den nod i openSet som har lägst fScore.
    def get_node_with_lowest_fscore(self, openSet, fScore):
        minCoords = None
        minScore = math.inf

        for node in openSet:
            for fNode, score in fScore:
                if fNode == node and score < minScore:
                    minScore = score
                    minCoords = node
        
        return minCoords

    # Tar in en koordinat och en karta och returnerar alla grannar till den koordinaten som är gångbara
    # (dvs. har värdet 0 i kartan).
    def get_neighbors(self, neighbors, stepCosts, coord):
        x, y = coord[0], coord[1]

        # Kollar alla möjliga håll (N, NÖ, Ö, SÖ, S, SV, V, NV)
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        numpy_map = numpy.array(self.map2D, dtype=numpy.uint8)
        
        # Apply threshold: pixels above threshold are obstacles (1), below are free (0)
        binary_map = (numpy_map >= self.threshold).astype(numpy.uint8)
        bin_map = 255 * binary_map
        
        #convert back to python arrooy
        maze = numpy.array(bin_map).tolist()


        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx and nx < len(self.map2D) and 0 <= ny and ny < len(self.map2D[0]):
                if maze[ny][nx] == 0:  # 1 är en fri cell
                    neighbors.append((nx, ny))
                else:
                    continue
            if (dx, dy) == (0, 1) or (dx, dy) == (1, 0) or (dx, dy) == (0, -1) or (dx, dy) == (-1, 0):
                stepCosts.append(1)
            elif (dx, dy) == (1, 1) or (dx, dy) == (1, -1) or (dx, dy) == (-1, -1) or (dx, dy) == (-1, 1):
                stepCosts.append(math.sqrt(dx**2 + dy**2))



    # A* funktionen som tar in en startkoordinat, en målkoordinat och en karta i form av en 2D lista
    # med ettor och nollor och räknar ut den kortaste vägen på kartan från start till mål med hjälp
    # av A* algoritmen.
    def a_star(self, start, goal):
        openSet = [start]  # Lista med hittade koordinater som kan behöva undersökas
        cameFrom = []  # Lista med tuples: (coord, previous_coord)
        gScore = []  # Lista med tuples: (coord, gscore)
        fScore = []  # Lista med tuples: (coord, fscore)

        self.set_score(gScore, start, 0)
        self.set_score(fScore, start, self.heuristic(start, goal))

        

        while openSet:
            current = self.get_node_with_lowest_fscore(openSet, fScore)
            
            if current == goal:
                path = []
                self.reconstruct_path(cameFrom, current, path)
                self.print_map_cv2(start, goal, path, cameFrom)

                multi_array_path = self.multi_array_constructor(path)

                self.path_list_pub.publish(multi_array_path )
            
            openSet.remove(current)
            
            neighbors = []
            stepCosts = []
            stepCostIdx = 0
            self.get_neighbors(neighbors, stepCosts, current)
            
            for neighbor in neighbors:
                tentativeGScore = self.get_score(gScore, current) + stepCosts[stepCostIdx]
                stepCostIdx += 1
                if tentativeGScore < self.get_score(gScore, neighbor):
                    # print('in if 1')
                    # Denna väg är bättre än tidigare känd väg, uppdatera vägen
                    for coord, previous in cameFrom:
                        # print('in for 2')
                        # Ta bort tidigare koordinat om den finns i cameFrom
                        if coord == neighbor:
                            # print('in if 2')
                            cameFrom.remove((coord, previous))
                            break

                    cameFrom.append((neighbor, current))
                    self.set_score(gScore, neighbor, tentativeGScore)
                    self.set_score(fScore, neighbor, tentativeGScore + self.heuristic(neighbor, goal))

                    if neighbor not in openSet:
                        # print('Adding neighbor to openSet')
                        openSet.append(neighbor)
        
        return None  # Ingen väg hittades

    def print_map_cv2(self, robot_pose, goal_pose, path, cameFrom):
        """
        Visualize a 2D map with robot and goal positions using OpenCV.

        Parameters:
        - map2D: 2D list or numpy array representing the map (0 = free space, 1 = obstacle)
        - robot_pose: tuple (x, y) for robot position
        - goal_pose: tuple (x, y) for goal position
        """
        # Convert map to numpy array if it isn't already
        map2D = numpy.array(self.map2D, dtype=numpy.uint8)

        # Apply threshold: pixels above threshold are obstacles (1), below are free (0)
        binary_map = (map2D >= self.threshold).astype(numpy.uint8)

        # Normalize map to 0-255 for visualization (0 = white, 1 = black obstacle)
        # Invert so free space is white, obstacles are black
        img = 255 * (1 - binary_map)

        # Convert single channel to BGR for colored markings
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


        for coord, _ in cameFrom:
            cv2.rectangle(img_color, coord, coord, color=(127, 127, 0), thickness=-1)
                
        for coord in path:
            cv2.rectangle(img_color, coord, coord, color=(127, 0, 127), thickness=-1)


        # Draw origin as a red circle
        cv2.circle(img_color, (0,0), radius=2, color=(0, 0, 255), thickness=-1)

        # Draw robot position as a blue circle
        cv2.circle(img_color, robot_pose, radius=2, color=(255, 0, 0), thickness=-1)

        # Draw goal position as a green circle
        cv2.circle(img_color, goal_pose, radius=2, color=(0, 255, 0), thickness=-1)

        # Optionally resize for better visibility
        scale = 4
        img_resized = cv2.resize(img_color, (img_color.shape[1]*scale, img_color.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        # Show the image
        cv2.imshow('Map Visualization', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    turtlebot3_a_star_node = Turtlebot3AStar()
    rclpy.spin(turtlebot3_a_star_node)

    turtlebot3_a_star_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
