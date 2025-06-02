#!/usr/bin/env python3
#

from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
import math


class Turtlebot3AStar(Node):

    def __init__(self):
        super().__init__('turtlebot3_a_star_node')
        
        self.a_star_map = UInt8MultiArray()
        self.start_goal_coords = UInt8MultiArray()

        self.map_recieved = False
        self.coords_recieved = False

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
            map2D = self.multi_array_deconstructor(self.a_star_map)
            coords = self.multi_array_deconstructor(self.start_goal_coords)
            #self.a_star(coords[0], coords[1], map2D)
            self.a_star((38,100), (25,75), map2D)

    def multi_array_deconstructor(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size

        # Reconstruct the 2D array from flat data list
        data = msg.data
        array_2d = [data[i * cols:(i + 1) * cols] for i in range(rows)]

        print(f"Deconstructed array:\n{array_2d=}")
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
    def reconstruct_path(self, cameFrom, current):
        totalPath = [current]

        # Bygger tillbaka vägen från målet till start
        while True:
            for node in cameFrom:
                if node[0] == current:
                    previous = node[1]
                    break
                else:
                    previous = None
            
            if previous == None:
                break
            
            current = previous
            totalPath.insert(0, current)
        
        print('Yay!')

        return totalPath

    # Heuristikfunktionen som tar in två koordinater, a och b, och beräknar avståndet mellan de.
    def heuristic(self, a, b):
        # Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Tar in en av scorlistorna, gScore eller fScore, en koordinat och ett nytt värde som den
    # koordinaten ska få och uppdaterar eller lägger till den i listan.
    def set_score(self, score_list, coord, newScore):
        for i in range(len(score_list)):
            if score_list[i][0] == coord:
                score_list[i] = (coord, newScore)
                return

        score_list.append((coord, newScore))

    # Tar in en av scorlistorna, gScore eller fScore och en koordinat och returnerar värdet som den
    # koordinaten har i listan. Om den inte finns returneras oändligheten.
    def get_score(self, score_list, coord):
        newScore = math.inf

        for node, score in score_list:
            if node == coord:
                newScore = score + 1

        return newScore

    # Tar in openSet listan och fScore listan och returnerar den nod i openSet som har lägst fScore.
    def get_node_with_lowest_fscore(self, openSet, fScore):
        minCoords = None
        minScore = math.inf

        for node in openSet:
            for score in fScore:
                if score[0] == node and score[1] < minScore:
                    minScore = score[1]
                    minCoords = node
        
        return minCoords

    # Tar in en koordinat och en karta och returnerar alla grannar till den koordinaten som är gångbara
    # (dvs. har värdet 0 i kartan).
    def get_neighbors(self, coord, maze):
        x, y = coord[0], coord[1]
        neighbors = []

        # Kollar alla möjliga håll (N, NÖ, Ö, SÖ, S, SV, V, NV)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Kollar om koordinaten är inom kartans gränser
            if nx < 0:
                nx = 0
            elif nx >= len(maze):
                nx = len(maze) - 1
            
            if ny < 0:
                ny = 0
            elif ny >= len(maze[0]):
                ny = len(maze[0]) - 1
            
            print(f"Checking neighbor: ({nx}, {ny})")
            print(f'Maze value: {len(maze)=}, {len(maze[0])=}')
            if nx == self.start_goal_coords.data[1][0]:
                input()
            
            if maze[nx][ny] == 0:  # 0 är en fri cell
                neighbors.append((nx, ny))
        
        return neighbors

    # A* funktionen som tar in en startkoordinat, en målkoordinat och en karta i form av en 2D lista
    # med ettor och nollor och räknar ut den kortaste vägen på kartan från start till mål med hjälp
    # av A* algoritmen.
    def a_star(self, start, goal, maze):
        openSet = [start]  # Lista med hittade koordinater som kan behöva undersökas
        cameFrom = []  # Lista med tuples: (coord, previous_coord)
        gScore = []  # Lista med tuples: (coord, gscore)
        fScore = []  # Lista med tuples: (coord, fscore)

        self.set_score(gScore, start, 0)
        self.set_score(fScore, start, self.heuristic(start, goal))

        while openSet:
            current = self.get_node_with_lowest_fscore(openSet, fScore)

            if current == goal:
                path = self.reconstruct_path(cameFrom, current)
                self.path_list_pub.publish(path)
            
            openSet.remove(current)

            for neighbor in self.get_neighbors(current, maze):
                tentativeGScore = self.get_score(gScore, current) + 1
                if tentativeGScore < self.get_score(gScore, neighbor):
                    # Denna väg är bättre än tidigare känd väg, uppdatera vägen
                    for coord, previous in cameFrom:
                        # Ta bort tidigare koordinat om den finns i cameFrom
                        if coord == neighbor:
                            cameFrom.remove((coord, previous))
                            break
                    cameFrom.append((neighbor, current))

                    self.set_score(gScore, neighbor, tentativeGScore)

                    self.set_score(fScore, neighbor, tentativeGScore + self.heuristic(neighbor, goal))

                    if neighbor not in openSet:
                        openSet.append(neighbor)
        
        return None  # Ingen väg hittades


def main(args=None):
    rclpy.init(args=args)
    turtlebot3_a_star_node = Turtlebot3AStar()
    rclpy.spin(turtlebot3_a_star_node)

    turtlebot3_a_star_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
