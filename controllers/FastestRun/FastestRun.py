'''COPY STARTS HERE'''
from controller import Robot, Camera
import time
import math
import numpy as np
from collections import deque
import cv2 as cv
from enum import Enum, IntEnum
from multiprocessing.connection import Client
from Utils import *
import json
import threading as thr
cos, sin, tan, radians, degrees, arccos, arcsin, arctan = np.cos, np.sin, np.tan, np.radians, np.degrees, np.arccos, np.arcsin, np.arctan2
PI = 3.14159
METERS_TO_INCHES = 39.3700787
EPSILON = 1e-7
UKNOWN_POS = (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0) # AAA (some weird values that represent unknown position)
ROBOT_SERVER = ('localhost', 31313)     # family is deduced to be 'AF_INET'

Search_Thread = None
Running_Search = False
Needs_Read_Value = False
Return_Search_Value = []
Search_Lock = thr.Lock()
def triangulate(x1, y1, a1, x2, y2, a2, x3, y3, a3):
    # Convert angles to radians
    a1 = math.radians(a1)
    a2 = math.radians(a2)
    a3 = math.radians(a3)

    # Calculate side lengths using law of sines
    if a1 + a2 + a3 != 180:
        raise ValueError("Angles do not add up to 180 degrees")
    elif a1 == 0 or a2 == 0 or a3 == 0:
        raise ValueError("Angle cannot be 0 degrees")
    else:
        # Calculate the distance between the three points
        d12 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        d23 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        d31 = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Use law of sines to calculate the angles between the points
        alpha = math.asin(d23 * math.sin(a1) / d31)
        beta = math.asin(d31 * math.sin(a2) / d12)
        gamma = math.pi - alpha - beta

        # Use trigonometry to calculate the distances from the points to your position
        p1 = d31 * math.sin(beta) / math.sin(gamma)
        p2 = d12 * math.sin(alpha) / math.sin(gamma)
        p3 = d23 * math.sin(math.pi - a3 - alpha) / math.sin(gamma)

        # Calculate your position by averaging the positions calculated from each point
        x = (p1 * x1 + p2 * x2 + p3 * x3) / (p1 + p2 + p3)
        y = (p1 * y1 + p2 * y2 + p3 * y3) / (p1 + p2 + p3)
    return (x, y)

class Vector2:
    '''x and y components OR the length and angle from X-Axis Counter Clockwise in Degrees'''
    def __init__(self,x=0, y=0, r=0, theta=0):
      if x!=0 or y!=0:
        self.x = x
        self.y = y
        self.r = ((self.x**2 + self.y**2)**0.5)
        self.theta = degrees(arctan(self.y,self.x))
      else:
        self.r = r
        self.theta = theta
        self.x = self.r * cos(radians(theta))
        self.y = self.r * sin(radians(theta))
        
    def plus(a, b) -> 'Vector2':
      return Vector2(x=a.x+b.x, y=a.y+b.y)
    def minus(a, b) -> 'Vector2':
          return Vector2(x=a.x-b.x, y=a.y-b.y)  
    def dot(self, b):
      return (self.x*b.x) + (self.y*b.y)
    def unit(self) -> 'Vector2':
      return Vector2(x=self.x/self.r, y=self.y/self.r) 
    def scale(self, scalar: float) -> 'Vector2':
      return Vector2(x=self.x*scalar, y=self.y*scalar)   
    def angle_from_dot(a, b):
      return degrees(arccos((a.dot(b)) / (a.r * b.r) ))
    def __str__(self):
      return "i:{}, j:{}, r:{}, theta:{}".format(self.x, self.y, self.r, self.theta)
    def __repr__(self):
      return "i:{}, j:{}, r:{}, theta:{}".format(self.x, self.y, self.r, self.theta)
    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        else:
            return None

class LandmarkTriangulator:
    class Color(IntEnum):
            RED = 0
            GREEN = 1
            BLUE = 2
            YELLOW = 3
    @staticmethod
    def to_color(color):
            return {0: LandmarkTriangulator.Color.RED, 1: LandmarkTriangulator.Color.GREEN, 2: LandmarkTriangulator.Color.BLUE, 3: LandmarkTriangulator.Color.YELLOW}[color]
    @staticmethod
    def get_color(color_vec):
            color_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
            sub_color = color_matrix - color_vec
            return LandmarkTriangulator.to_color(np.argmin(np.sum(sub_color**2, axis=1)))
    class IdentifiedObject:
        def __init__(self, d_pos:'Vector2', color):
            self.d_pos = d_pos
            self.color = color
    class Landmark:
        def __init__(self, x, y, color):
            self.pos = Vector2(x=x, y=y)
            self.color = color
    def __init__(self):
        self.landmarks = {}
        self.object_buffer = []
        self.identified_objects = 0
    def clearBuffer(self):
        self.object_buffer = []
        self.identified_objects = 0
    def addLandmark(self, x, y, color):
        self.landmarks[color] = LandmarkTriangulator.Landmark(x=x, y=y, color=color)
    def storeIdentifiedObject(self, obj):
        self.identified_objects += 1
        self.object_buffer.append(obj)
    def hasColor(self, color):
        for l in self.object_buffer:
            if l.color == color:
                return True
        return False
    def getLandmarkData(self): # Stores all of the necessary data for Triangulation
        landmark_data = {}
        if self.identified_objects < 3:
            #print("Not enough identified objects to triangulate")
            return None
        for i in range(3):
            obj = self.object_buffer[i]
            landmark_data[obj.color] = {'ref': self.landmarks[obj.color].pos, 'obj': obj.d_pos}
        return landmark_data
    def trilateration(point_a, point_b, point_c, distance_a, distance_b, distance_c):
        ax, ay = point_a
        bx, by = point_b
        cx, cy = point_c

        eq1_lhs = (ax - bx) * 2, (ay - by) * 2
        eq1_rhs = (distance_b ** 2) - (distance_a ** 2) + (ax ** 2) - (bx ** 2) + (ay ** 2) - (by ** 2)

        eq2_lhs = (bx - cx) * 2, (by - cy) * 2
        eq2_rhs = (distance_c ** 2) - (distance_b ** 2) + (bx ** 2) - (cx ** 2) + (by ** 2) - (cy ** 2)

        A = np.array([eq1_lhs, eq2_lhs])
        B = np.array([eq1_rhs, eq2_rhs])

        position_x, position_y = np.linalg.solve(A, B)
        return position_x, position_y
    def Triangulate(self):
        landmark_data = self.getLandmarkData()
        c1, c2, c3 = None, None, None
        for k,v in landmark_data.items():
            if c1 is None:
                c1 = v
            elif c2 is None:
                c2 = v
            elif c3 is None:
                c3 = v               
        A = -2*c1['ref'].x + 2*c2['ref'].x
        B = -2*c1['ref'].y + 2*c2['ref'].y
        C = c1['obj'].r**2 - c2['obj'].r**2 -c1['ref'].x**2 + c2['ref'].x**2 - 2*c1['ref'].y**2 + c2['ref'].y**2
        D = -2*c2['ref'].x + 2*c3['ref'].x
        E = -2*c2['ref'].y + 2*c3['ref'].y
        F = c2['obj'].r**2 - c3['obj'].r**2 -c2['ref'].x**2 + c3['ref'].x**2 - 2*c2['ref'].y**2 + c3['ref'].y**2
        try:
            x = (C*E - F*B) / ((E*A - B*D))
            y = (C*D - A*F) / ((B*D - A*E))
        except ZeroDivisionError:
            return None
        #print(Vector2(x=x, y=y))
        return Vector2(x=x, y=y)
    def canTriangulate(self):
        return self.identified_objects >= 3

# For this lab we use a non probabalistic model for the Grid.
class Grid:
    BOUNDS = 16
    LIDAR_MAX_RNG = 1 # Meter
    LIDAR_MIN_RNG = 0.01
    WORLD_SIZE = 2.88 * METERS_TO_INCHES
    CELL_SIZE = WORLD_SIZE / BOUNDS
    
    class Node:
        class Edge:
            class Type(IntEnum):
                NORTH = 0
                SOUTH = 1
                EAST = 2
                WEST = 3
                @staticmethod
                def negation(edge_type):
                    return {
                        Grid.Node.Edge.Type.NORTH: Grid.Node.Edge.Type.SOUTH,
                        Grid.Node.Edge.Type.SOUTH: Grid.Node.Edge.Type.NORTH,
                        Grid.Node.Edge.Type.EAST: Grid.Node.Edge.Type.WEST,
                        Grid.Node.Edge.Type.WEST: Grid.Node.Edge.Type.EAST
                    }[edge_type]
                @staticmethod
                def getDirection(dir):
                    return {1: Grid.Node.Edge.Type.NORTH, 3: Grid.Node.Edge.Type.SOUTH, 0: Grid.Node.Edge.Type.EAST, 2: Grid.Node.Edge.Type.WEST}[dir]
                @staticmethod
                def __plus__(a, b): # Add the Theta Repr, Modulo 360, get Cardinal Direction, convert back to Type 
                    return Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection((Grid.Node.Edge.Type.getTheta(a) + Grid.Node.Edge.Type.getTheta(b)) % 360))
                @staticmethod
                def getTheta(dir) -> int:
                    return {Grid.Node.Edge.Type.NORTH: 90, Grid.Node.Edge.Type.SOUTH: 270, Grid.Node.Edge.Type.EAST: 0, Grid.Node.Edge.Type.WEST: 180}[dir]    
                @staticmethod
                def toVector(dir):
                    return {Grid.Node.Edge.Type.NORTH: Vector2(x=-1, y=0),  Grid.Node.Edge.Type.SOUTH: Vector2(x=1, y=0), Grid.Node.Edge.Type.EAST: Vector2(x=0, y=1), Grid.Node.Edge.Type.WEST: Vector2(x=0, y=-1)}[dir]
            class Weighting(IntEnum):
                FREE = 1
                BARRIER = 0
            def __init__(self, origin, to, type):
                self.origin = origin
                self.to = to
                self.type = type
                self.weighting = Grid.Node.Edge.Weighting.FREE
            def __str__(self):
                if self.weighting == Grid.Node.Edge.Weighting.FREE:
                    return f"{self.to.grid_cell_number()}"
                else:
                    return "X"
        def __init__(self, x=-1, y=-1):
            self.x = x
            self.y = y
            self.edges = {
                Grid.Node.Edge.Type.NORTH: None,
                Grid.Node.Edge.Type.SOUTH: None,
                Grid.Node.Edge.Type.EAST: None,
                Grid.Node.Edge.Type.WEST: None
            }
        def pos(self):
            return Vector2(x=int(self.x), y=int(self.y))
        def neighbors(self):
            return [node.to for node in self.edges.values() if node is not None]
        def grid_cell_number(self):
            return (self.x*Grid.BOUNDS) + (self.y+1)
        @staticmethod
        def link(node1, node2, dir1, dir2):
            node1.edges[dir1] = Grid.Node.Edge(node1, node2, dir2)
            node2.edges[dir2] = Grid.Node.Edge(node2, node1, dir1)
        @staticmethod
        def h_distance(start_node, end_node):
            return ((start_node.x - end_node.x)**2 + (start_node.y - end_node.y)**2)**0.5
        @staticmethod
        def h_travel(start_node, path):
            return sum([Grid.Node.h_distance(start_node, end_node) for end_node in path.path])
        @staticmethod
        def h_actions_to(start_node, end_node, current_dir: Edge.Type):
            curr_node = start_node
            num_actions = 0
            #TODO: Implement this
            
        def __str__(self):
            s = f"Cell {self.grid_cell_number()}\n"
            s += f"North: {self.edges[Grid.Node.Edge.Type.NORTH]}\n"
            s += f"South: {self.edges[Grid.Node.Edge.Type.SOUTH]}\n"
            s += f"East:  {self.edges[Grid.Node.Edge.Type.EAST]}\n"
            s += f"West:  {self.edges[Grid.Node.Edge.Type.WEST]}"
            return s
        ''' This Function returns the relationship between the two nodes w.r.t node a to node b, or None if there is no relationship. '''
        @staticmethod
        def relationship(node_a, node_b):
            for edge in node_a.edges.values():
                if edge is not None and edge.to == node_b:
                    return Grid.Node.Edge.Type.negation(edge.type), edge.weighting
            return None, None
    class GridGenerator:
        def __init__(self):
            self.current_cell = Grid.Node(0,0)
            self.visited_cells = set()
            self.visited_cells.add(self.current_cell)
            self.discovered_cells = {}
            self.discovered_positions = {} # Maps a Node to a Position
            self.discovered_cells[(0,0)] = self.current_cell
            self.depth_memory = []
        def __getitem__(self, indecies):
            i, j = indecies
            return self.discovered_cells[(i,j)]
            
        def unvisited_cells(self):
            return [cell for cell in self.discovered_cells.values() if cell not in self.visited_cells]
        def update_current_pos(self, pos):
            self.discovered_positions[self.current_cell] = pos
        def update_cells_in_direction(self, current_cell, dir, num_cells):
            if num_cells == 0:
                return
            dif = Grid.Node.Edge.Type.toVector(dir)
            current_pos = Vector2(x=current_cell.x, y=current_cell.y).plus(dif)
            cell_in_dir = None
            try:
                cell_in_dir = self.discovered_cells[(int(current_pos.x), int(current_pos.y))]
            except KeyError:
                cell_in_dir = None
            if num_cells > 0: # Cells are Discovered
                if cell_in_dir == None: # We haven't found this Cell Yet
                    cell_in_dir = Grid.Node(current_pos.x, current_pos.y)
                    self.discovered_cells[(int(current_pos.x), int(current_pos.y))] = cell_in_dir
                else:
                    Grid.Node.link(current_cell, cell_in_dir, dir, Grid.Node.Edge.Type.negation(dir))
                self.update_cells_in_direction(cell_in_dir, dir, num_cells - 1)
            else: # Lets Check if we have found a Barrier
                dif = Grid.Node.Edge.Type.toVector(dir)
                current_pos = Vector2(x=current_cell.x, y=current_cell.y).plus(dif)
                try:
                    if cell_in_dir!= None: # We found a Barrier, Make sure those nodes are linked
                        Grid.Node.link(current_cell, cell_in_dir, dir, Grid.Node.Edge.Type.negation(dir))
                        current_cell.edges[dir].weighting = Grid.Node.Edge.Weighting.BARRIER
                        cell_in_dir.edges[Grid.Node.Edge.Type.negation(dir)].weighting = Grid.Node.Edge.Weighting.BARRIER
                        #print(f"Found Barrier...")
                except KeyError:
                    pass # We didn't find a Barrier, so we can't update the current cell
        def generate_grid(self, lidar_img, current_theta):
            lidar_img = Grid.convertLidarImg(lidar_img)
            self.depth_memory.append((self.current_cell, lidar_img, current_theta))
            look_direction = Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(current_theta))
            # look_theta = Grid.Node.Edge.Type.getTheta(look_direction)            
            unobstructed_cells_forward, unobstructed_cells_left, unobstructed_cells_behind, unobstructed_cells_right = min(lidar_img[0] // Grid.CELL_SIZE, 100), min(lidar_img[270] // Grid.CELL_SIZE, 100), min(lidar_img[180] // Grid.CELL_SIZE, 100), min(lidar_img[90] // Grid.CELL_SIZE, 100) #Inches
            cell_map = {
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.EAST): unobstructed_cells_forward, 
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.NORTH): unobstructed_cells_left,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.WEST): unobstructed_cells_behind,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.SOUTH): unobstructed_cells_right,
            }
            for dir, num_cells in cell_map.items():
                self.update_cells_in_direction(self.current_cell, dir, num_cells)
            # for cell in self.discovered_cells.values():
            #     for direction in [Grid.Node.Edge.Type.EAST, Grid.Node.Edge.Type.WEST, Grid.Node.Edge.Type.NORTH, Grid.Node.Edge.Type.SOUTH]:
            #         pos = Vector2(x=cell.x, y=cell.y).plus(Grid.Node.Edge.Type.toVector(direction))
            #         if (int(pos.x), int(pos.y)) in self.discovered_cells:
            #             connect_node = self.discovered_cells[(int(pos.x), int(pos.y))]
            #             Grid.Node.link(cell, connect_node, direction, Grid.Node.Edge.Type.negation(direction))
                
    def __init__(self, use_grid_generator=False):
        self.nodes = {}
        self.visited_cells = set()
        self.current_cell = None
        self._lastKnownPosition = None
        self.grid_generator = Grid.GridGenerator()
        self.previous_actions = deque()
        self._lastKnownCell = None
        self.UseGridGenerator = use_grid_generator
        self.destination_cell = None
        self.discovered_cells = deque()
        
        for i in range(Grid.BOUNDS): # Create the grid of Nodes
            for j in range(Grid.BOUNDS):
                self.nodes[(i,j)] = Grid.Node(i,j)
        for i in range(Grid.BOUNDS): # Connect the grid of Nodes
            for j in range(Grid.BOUNDS):
                if i > 0:
                    Grid.Node.link(self[i,j], self[i-1,j], Grid.Node.Edge.Type.NORTH, Grid.Node.Edge.Type.SOUTH)                    
                if i < Grid.BOUNDS-1:
                    Grid.Node.link(self[i,j], self[i+1,j], Grid.Node.Edge.Type.SOUTH, Grid.Node.Edge.Type.NORTH)   
                if j > 0:
                    Grid.Node.link(self[i,j], self[i,j-1], Grid.Node.Edge.Type.WEST, Grid.Node.Edge.Type.EAST)   
                if j < Grid.BOUNDS-1:
                    Grid.Node.link(self[i,j], self[i,j+1], Grid.Node.Edge.Type.EAST, Grid.Node.Edge.Type.WEST)   
    
    def set_destination_cell(self, cell_num):
        for node in self.nodes.values():
            if node.grid_cell_number() == cell_num:
                self.destination_cell = node
    @staticmethod
    def save_to_file(grid, file_path):
        grid_data = {}
        for (x, y), node in grid.nodes.items():
            edges_data = {}
            for edge_type, edge in node.edges.items():
                if edge is not None:
                    edges_data[edge_type.name] = {
                        'to': f"{edge.to.x},{edge.to.y}",
                        'weighting': edge.weighting.name
                    }
            grid_data[f"{x},{y}"] = edges_data

        visited_ids = [f"{node.x},{node.y}" for node in grid.visited_cells]

        data = {
            'grid': grid_data,
            'visited': visited_ids
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        grid_data = data['grid']
        visited_cells = data['visited']

        grid = Grid()

        for node_coords, edges_data in grid_data.items():
            x, y = map(int, node_coords.split(','))
            for edge_type_name, edge_data in edges_data.items():
                edge_type = Grid.Node.Edge.Type[edge_type_name]
                to_x, to_y = map(int, edge_data['to'].split(','))
                weighting = Grid.Node.Edge.Weighting[edge_data['weighting']]

                if grid.nodes[(x, y)].edges[edge_type] is None:
                    Grid.Node.link(grid.nodes[(x, y)], grid.nodes[(to_x, to_y)], edge_type, Grid.Node.Edge.Type.negation(edge_type))
                
                grid.nodes[(x, y)].edges[edge_type].weighting = weighting
                grid.nodes[(to_x, to_y)].edges[Grid.Node.Edge.Type.negation(edge_type)].weighting = weighting

        grid.visited_cells = {grid.nodes[tuple(map(int, node_pos.split(',')))] for node_pos in visited_cells}
        #print('Visited Cells: ', grid.visited_cells)

        return grid

    def set_current_cell_pos(self, x, y):
        self.current_cell = self[x, y]
        self.visited_cells.add(self.current_cell)
        #print('Setting Current Cell to: ', self.current_cell.grid_cell_number())
        
    def set_current_cell(self, grid_cell_number):
        for node in self.nodes.values():
            if node.grid_cell_number() == grid_cell_number:
                self.current_cell = node
                self.visited_cells.add(self.current_cell)
                #print('Setting Current Cell to:', self.current_cell.grid_cell_number())
                if self.UseGridGenerator:
                    self.grid_generator.update_current_pos(self.current_cell.pos())
                
    def __getitem__(self, indecies):
        i, j = indecies
        return self.nodes[(i,j)]
    
    def travel(self, dir, seq_actions=1):
        current_cell = self.get_current_cell()
        #print('lol')
        #print(current_cell, dir)
        for _ in range(seq_actions):
            if current_cell.edges[dir] != None and current_cell.edges[dir].weighting != Grid.Node.Edge.Weighting.BARRIER:
                current_cell = current_cell.edges[dir].to
                if self.UseGridGenerator:
                    self.grid_generator.visited_cells.add(current_cell)
                    self.grid_generator.current_cell = current_cell
                else:
                    self.visited_cells.add(current_cell)
                    self.current_cell = current_cell
            else:
                return False
        return True
    
    def rec_follow_prob(self, current_cell, prob, temperature, depth=0, max_depth=5):
        if depth == max_depth:
            return prob * temperature * 0.8 * (1-depth/5)
        for dir, weighting in current_cell.edges.items():
            if weighting is not None:
                if weighting.to is not None and weighting.to not in self.visited_cells:
                    return prob * temperature * 0.8 * (1-depth/5) * self.rec_follow_prob(weighting.to, prob=prob, temperature=temperature, depth=depth+1)
                elif weighting.weighting == Grid.Node.Edge.Weighting.BARRIER:
                    return 1
                else:
                    return 1
        return 1
        
    def follow_prob(self, temperature):
        current_cell = self.get_current_cell()
        return 1 - self.rec_follow_prob(current_cell=current_cell, prob=1, temperature=temperature, depth=0, max_depth=3)
        
    
    def unvisitedCells(self):
        if not self.UseGridGenerator:
            return [cell for cell in self.nodes.values() if cell not in self.visited_cells]
        else:
            return self.grid_generator.unvisited_cells()
    
    def location_known(self):
        return self.current_cell != None
    
    @staticmethod
    def updateGraphRelationship(grid:'Grid', grid_generator:'Grid.GridGenerator', current_cell:'Grid.Node', current_pos:'Vector2', updated_cells= []):
        if current_cell not in grid_generator.discovered_cells.values() or current_cell in updated_cells: # BASE CASE
            return
        if current_cell in grid_generator.visited_cells: # Update Visited Status
            grid.visited_cells.add(grid[current_pos.x, current_pos.y])
        cell_in_grid = grid[current_pos.x, current_pos.y]
        needs_update = []
        for neighbor in current_cell.neighbors():
            dir, weighting = Grid.Node.relationship(current_cell, neighbor)
            pos_of_neighbor = current_pos.plus(Grid.Node.Edge.Type.toVector(dir))
            cell_in_grid.edges[dir].weighting = weighting
            cell_in_grid.edges[dir].to.edges[Grid.Node.Edge.Type.negation(dir)].weighting = weighting
            needs_update.append((neighbor, pos_of_neighbor))
        updated_cells.append(current_cell)
        for neighbor, pos_of_neighbor in needs_update:
            grid_generator.discovered_positions[(pos_of_neighbor.x, pos_of_neighbor.y)] = neighbor
            Grid.updateGraphRelationship(grid, grid_generator, neighbor, pos_of_neighbor, updated_cells)            

    def translate_grid_generator(self):
        # Here we want to take all of the Nodes in the GridGenerator and translate them to known Cell Positions. 
        if len(self.grid_generator.discovered_positions.values()) > 0:
            #print(list(self.grid_generator.discovered_positions.items()))
            node, pos = list(self.grid_generator.discovered_positions.items())[0]
            # Some node in our Grid Generator and then find the paths traveled by the grid generator and translate them into the cells of the grid
            new_grid = Grid(use_grid_generator=True)
            Grid.updateGraphRelationship(new_grid, self.grid_generator, node, pos, [])
            new_grid.grid_generator = self.grid_generator
            return new_grid
        else:
            return None
    def get_current_cell(self):
        return self.current_cell if not self.UseGridGenerator else self.grid_generator.current_cell
      
    def find_path_to(self, end):
        # perform breadth-first search to find the shortest path from start to end
        start = self.get_current_cell()
        q = deque()
        q.append(start)
        visited = set()
        parent = {}

        while len(q) > 0:
            curr_node = q.popleft()
            visited.add(curr_node)

            if curr_node == end:
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path

            for edge_type, edge in curr_node.edges.items():
                if edge and edge.weighting != Grid.Node.Edge.Weighting.BARRIER:
                    next_node = edge.to
                    if next_node not in visited and next_node not in q:
                        q.append(next_node)
                        parent[next_node] = curr_node
    
        return []
    # Lidar Needs to be converted to Inches, 
    @staticmethod
    def convertLidarImg(lidar_img):
        return np.array(lidar_img) * METERS_TO_INCHES
    
    def getNumCellsInDirection(self, dir, starting_cell=None):
        curr_cell = self.current_cell if starting_cell == None else starting_cell
        for i in range(int(4)):
            if curr_cell != None:
                curr_cell = curr_cell.edges[dir]
                if curr_cell != None:
                    if curr_cell.weighting == Grid.Node.Edge.Weighting.BARRIER:
                        break
                    curr_cell = curr_cell.to
        return i
    def getCellInDirection(self, dir, num_cells, starting_cell=None):
        if num_cells == -1:
            return None
        curr_cell = self.current_cell if starting_cell == None else starting_cell
        if num_cells > 4:
            return None
        for i in range(int(num_cells)):
            if curr_cell != None:
                if curr_cell.grid_cell_number() == 161:
                    #print('ISSUE EIISUDAU')
                    print(num_cells, )
                old_cell = curr_cell
                curr_cell = curr_cell.edges[dir]
                if curr_cell!= None:
                    if i < num_cells:
                        self.discovered_cells.appendleft(curr_cell.to)
                        curr_cell.weighting = Grid.Node.Edge.Weighting.FREE
                        curr_cell.to.edges[Grid.Node.Edge.Type.negation(dir)].weighting = Grid.Node.Edge.Weighting.FREE

                    curr_cell = curr_cell.to
        return curr_cell
    def updateBarrier(self, cell, dir):
        if cell and cell.edges[dir] != None:
            cell.edges[dir].weighting = Grid.Node.Edge.Weighting.BARRIER
            cell.edges[dir].to.edges[Grid.Node.Edge.Type.negation(dir)].weighting = Grid.Node.Edge.Weighting.BARRIER
    def update_barriers_from_memory(self):
        for ref_cell, lidar_img, current_theta in self.grid_generator.depth_memory:
            grid_cell = None
            for pos, cell in self.grid_generator.discovered_positions.items(): # Find the Position associated with that reference
                if cell == ref_cell:
                    grid_cell = self[(pos[0], pos[1])]
            if grid_cell == None:
                continue
            look_direction = Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(current_theta))
            unobstructed_cells_forward, unobstructed_cells_left, unobstructed_cells_behind, unobstructed_cells_right = min(lidar_img[0] // self.CELL_SIZE, 100), min(lidar_img[270] // self.CELL_SIZE, 100), min(lidar_img[180] // self.CELL_SIZE, 100), min(lidar_img[90] // self.CELL_SIZE, 100) #Inches
            cell_map = {
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.EAST): unobstructed_cells_forward, 
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.NORTH): unobstructed_cells_left,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.WEST): unobstructed_cells_behind,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.SOUTH): unobstructed_cells_right,
            }
            for dir, num_cells in cell_map.items():
                update_cell = self.getCellInDirection(dir, num_cells, starting_cell=grid_cell)
                if update_cell:
                    self.updateBarrier(update_cell, dir)
    def get_last_discovered(self):
        self.discovered_cells = deque([cell for cell in self.discovered_cells if cell not in self.visited_cells])
        return self.discovered_cells
    
    def update_barriers(self, lidar_img, current_theta):
        ''' What we would want to Do is look into every cell as compared to our cell and update whether or not it has a barrier '''
        if self.UseGridGenerator:
            self.grid_generator.generate_grid(lidar_img, current_theta)
        else:
            lidar_img = self.convertLidarImg(lidar_img)
            look_direction = Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(current_theta))
            
            unobstructed_cells = []
            for angle in [0, 270, 180, 90]:
                distance = lidar_img[angle]
                if math.isnan(distance) or math.isinf(distance) or distance > Grid.CELL_SIZE*2:
                    #print('eiwfcewibcwheb')
                    unobstructed_cells.append(-1)
                else:
                    try:
                        unobstructed_cells.append(min(distance // self.CELL_SIZE, 100))
                    except ZeroDivisionError:
                        unobstructed_cells.append(-1)

            unobstructed_cells_forward, unobstructed_cells_left, unobstructed_cells_behind, unobstructed_cells_right = unobstructed_cells

            cell_map = {
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.EAST): unobstructed_cells_forward, 
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.NORTH): unobstructed_cells_left,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.WEST): unobstructed_cells_behind,
                Grid.Node.Edge.Type.__plus__(look_direction, Grid.Node.Edge.Type.SOUTH): unobstructed_cells_right,
            }
            for dir, num_cells in cell_map.items():
                update_cell = self.getCellInDirection(dir, num_cells)
                if update_cell and update_cell:
                    self.updateBarrier(update_cell, dir)
                    
    def get_actions(self,):
        actions = []
        current_cell = self.get_current_cell()
        for dir, edge in current_cell.edges.items():
            if edge != None and edge.weighting != Grid.Node.Edge.Weighting.BARRIER:
                actions.append(dir)
        return actions
    def __str__(self):
        rows = []
        for row in range(self.BOUNDS):
            col_strings = []
            for col in range(self.BOUNDS):
                node = self[row, col]
                col_strings.append(str(node).split('\n'))
            row_str = ""
            for i in range(len(col_strings[0])):
                for j in range(len(col_strings)):
                    if 'Cell' in col_strings[j][i]:
                        row_str += "\t" + col_strings[j][i] + "\t"
                    else:    
                        row_str += "\t" + col_strings[j][i] + "\t"
                row_str += "\n"
            row_str += "-" * 100 + "\n"
            rows.append(row_str)
        return "".join(rows)
    def print_visited_cells(self):
        if self.UseGridGenerator:
            for i in range(Grid.BOUNDS):
                for j in range(Grid.BOUNDS): 
                    if (i, j) in self.grid_generator.discovered_positions and self.grid_generator.discovered_positions[(i, j)] in self.grid_generator.visited_cells:
                        print("X", end="")
                    else:
                        print(".", end="")
                print()
        else:
            for i in range(Grid.BOUNDS):
                for j in range(Grid.BOUNDS):
                    if self[i, j] in self.visited_cells:
                        print("X", end="")
                    else:
                        print(".", end="")
                print()
    def get_current_grid_cell_number(self):
        if self.UseGridGenerator:
            pos = None
            for p, cell in self.grid_generator.discovered_positions.items():       
                if cell == self.grid_generator.current_cell:
                    pos = p
            if pos == None:
                raise KeyError
            if isinstance(pos, Vector2):
                return Grid.Node(pos.x, pos.y).grid_cell_number()
            else:
               return Grid.Node(pos[0], pos[1]).grid_cell_number() 
        else:
            return self.current_cell.grid_cell_number()
    @staticmethod
    def get_cell_from_position(position):
        #print('get_cell_from_position', position)
        try:
            # #print(position    )
            row = (position.x + 20) // Grid.CELL_SIZE
            row = np.clip(row, 0.0, Grid.BOUNDS - 1.0)
            col = (-position.y + 20) // Grid.CELL_SIZE
            col = np.clip(col, 0, Grid.BOUNDS - 1)
            #print('Row', row, 'Col', col)
            return col, row
        except:
            return None     
    def get_cell_by_id(self, cell_id):
        cell = [c for c in self.nodes if c.grid_cell_number() == cell_id]
        return cell if cell else None    
    def get_optimal_unvisited_exploration(self, current_dir: Node.Edge.Type):
            undiscovered = []
        # if front_cell!= None and front_cell.weighting!= Grid.Node.Edge.Weighting.BARRIER and front_cell.to not in self.visited_cells:
        #     return front_cell.to
        # else"""  """:
            for dir, edge in self.current_cell.edges.items():
                if edge!= None and edge.weighting!= Grid.Node.Edge.Weighting.BARRIER and edge.to not in self.visited_cells:
                    undiscovered.append(edge.to)
            # if len(undiscovered) == 0:
            #     return self.rec_unvisited_search(self.current_cell, self.destination_cell, ) # Only search when no other options are available
            else:
                #print('eidjwneidwkhjb')
                return np.random.choice(undiscovered)
    class Path:
        k_h1 = 0.8
        k_h2 = 0.4
        def __init__(self, init_node, goal_node):
            self.path = [init_node]
            self.current_weight = 0
            self.goal_node = goal_node 
            self.h = Grid.Node.h_distance(init_node, goal_node)
        @staticmethod
        def Copy(path):
            path_copy = Grid.Path(path.path[0], path.path[0])
            path_copy.goal_node = path.goal_node
            path_copy.path = []
            for i in range(len(path.path)):
                path_copy.path.append(path.path[i])
                path_copy.current_weight += 1
            path_copy.current_weight = path.current_weight
            return path_copy
        def cost(self):
            return self.h + self.current_weight
        def extend(self, new_node, grid):
            new_path = Grid.Path.Copy(self)
            new_path.path.append(new_node)
            self.h = self.k_h1*Grid.Node.h_distance(new_node, new_path.goal_node) + sum([0.5 for cell in new_path.path if cell not in grid.visited_cells]) # We encourage exploration 
            return new_path
        def __lt__(self, other):
                return self.cost() < other.cost()
        def __gt__(self, other):
                return self.cost() > other.cost()
        def __eq__(self, other):
            if self is None or other is None:
                return False
            return self.cost() == other.cost()
        def __neq__(self, other):
                return not (self == other)
            
    def call_rec_unvisited_search(self, start_cell: Node.Edge.Type, goal_cell: Node.Edge.Type, **kwargs):
        global Search_Lock, Search_Thread, Running_Search, Return_Search_Value, Needs_Read_Value
        paths_found = HeapQueue()
        path = self.find_path_to(goal_cell)
        with Search_Lock:
            Running_Search = False
            if not isinstance(Return_Search_Value, Grid.Path):
                Return_Search_Value = path
                # #print('setting val')
                #print(Return_Search_Value)
                Needs_Read_Value = True                
                return
            
    
    def rec_unvisited_search(self, start_cell: Node.Edge.Type, goal_cell: Node.Edge.Type, frontier=None, paths_found= None, stop_depth= 100, depth=0, max_frontier_size=-1):
        global Search_Lock, Search_Thread, Running_Search, Return_Search_Value, Needs_Read_Value
        if frontier == None:
            frontier = HeapQueue()
            frontier.push(Grid.Path(start_cell, goal_cell))
        while True:
            path = frontier.pop()
            if (len(path.path)) > stop_depth:
                paths_found.append(path)
            #print(p for p in path.path)
            if path.path[-1] == goal_cell or path.path[-1] not in self.visited_cells:
                #print('Ending Found', depth, frontier, )
                with Search_Lock:
                    Return_Search_Value = path
                    #print('setting val')
                    #print(Return_Search_Value)
                    Needs_Read_Value = True
                    return
            for dir, edge in path.path[-1].edges.items():
                if edge!= None and edge.weighting!= Grid.Node.Edge.Weighting.BARRIER and edge.to not in path.path:
                    frontier.push(path.extend(edge.to, self))
                    if max_frontier_size != -1 and frontier.size() >= max_frontier_size:
                        frontier.remove_worst() 
                    #print('Search')
                    depth+=1
                
    
    def rec_search(self, start_cell: Node.Edge.Type, goal_cell: Node.Edge.Type, frontier=None, stop_depth= 50, depth=0, max_frontier_size=-1):
        if frontier == None:
            frontier = HeapQueue()
            frontier.push(Grid.Path(start_cell, goal_cell))
        if depth == stop_depth:
            return frontier.pop()
        path = frontier.pop()
        if path.path[-1] == goal_cell:
            return path
        for dir, edge in path.path[-1].edges.items():
            if edge!= None and edge.weighting!= Grid.Node.Edge.Weighting.BARRIER and edge.to not in path.path:
                frontier.push(path.extend(edge.to, self))
                if max_frontier_size != -1 and frontier.size() >= max_frontier_size:
                    frontier.remove_worst() 
                found = self.rec_search(edge.to, goal_cell, frontier, depth=depth+1, stop_depth=stop_depth, max_frontier_size=max_frontier_size)
                if found:
                    return found
        #print('Ending Search', depth, frontier, frontier.pop())
        return None
         
# Class Which Performs the Calculations associated with Transitioning from One Grid Cell to Another Grid Cell
class TransitionModel:
    def __init__(self, agent, grid):
        self.agent = agent
        self.grid = grid
    def cntActionsInDir(self, dir, path):
        cnt = 1
        try:
            c_node = self.grid.current_cell.edges[dir].to
            for idx in range(len(path)):
                node = path[idx]
                if c_node.edges[dir] != None:
                    if c_node.edges[dir].to == node and c_node.edges[dir].weighting != Grid.Node.Edge.Weighting.BARRIER and (node in self.grid.get_last_discovered() or node in self.grid.visited_cells):
                        cnt += 1
                        c_node = node
                else:
                    break
            print('cnt,', cnt)
            for _ in range(cnt-1):
                path.popleft()
            return cnt
        except IndexError:
            return 1
    # Transitions Agent from one Grid Cell to another. If the cell is not blocked, transitions and returns True. Otherwise, returns False
    def TransitionState(self, action, path) -> bool:
        theta_target = Grid.Node.Edge.Type.getTheta(action)
        motion_turn = TurnTarget(theta_target, self.agent['theta'])
        # desired_pos = self.agent['current_pos'].plus(Vector2(r=, theta=theta_target))
        correction_motion = None
        if self.agent.cnt_movements > 0:
            if self.grid.current_cell.edges[Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(self.agent['theta']))] == None or self.agent['lidar'][0] < Grid.CELL_SIZE:
                correction_motion = self.agent.dest_deq.append(DistanceTarget(DistanceTarget.Side.FRONT, Grid.CELL_SIZE/2))
                self.agent.cnt_movements = 0
        cnt = self.cntActionsInDir(action, path)
        print(action, cnt,)
        if self.grid.travel(action, cnt):
            if not Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(self.agent['theta'])) == action:
                 self.agent.dest_deq.append(StopTarget())
                 self.agent.dest_deq.append(motion_turn)
            if correction_motion:
                self.agent.dest_deq.append(correction_motion)
            motion_forward = MotionTarget(self.agent.TotalDistanceTraveled, Grid.CELL_SIZE*cnt)
           
            self.agent.dest_deq.append(motion_forward)
            if self.grid.getNumCellsInDirection(action) < 2:
                 self.agent.dest_deq.append(StopTarget())
            self.agent.dest_deq.append(UpdateTarget())
            agent.cnt_movements+=1
            if len(self.grid.visited_cells) % 5 == 0:
                Grid.save_to_file(agent.grid, 'GridState.json')
            return True
        return False

class Target:
    class Type(Enum):
        POS = 0
        DISTANCE = 1
        WALL = 2
        TURN = 3 
        MOTION = 5
        TRI = 4
        STOP = 7
        UPDATE = 8
    def __init__(self, type):
        self.type = type
        self.action_time = 0
class TriangulateTarget(Target):
    def __init__(self,starting_angle):
        Target.__init__(self, Target.Type.TRI)
        self.starting_angle = starting_angle
        self.current_angle = self.starting_angle
        self.previous_angle = self.starting_angle
        self.total_angle = 0
    def update(self, reading):
        self.previous_angle = self.current_angle
        self.current_angle = reading
        self.total_angle += get_dTheta(self.current_angle, self.previous_angle)
    def revolutions(self):
        return abs(self.total_angle) / 360
class StopTarget(Target):
    def __init__(self):
        Target.__init__(self, Target.Type.STOP)
        
class DistanceTarget(Target):
    class Side(Enum):
        FRONT = 0
        REAR = 1
        LEFT = 2
        RIGHT = 3        
    def __init__(self, side, distance):
        Target.__init__(self, Target.Type.DISTANCE)
        self.side = side
        self.distance = distance
class TurnTarget(Target):
    def __init__(self, desired_theta:float, currentReading):
        Target.__init__(self, Target.Type.TURN)
        self.thetaTarget = desired_theta
        self._thetaError = get_dTheta(currentReading, self.thetaTarget)
        self._prevThetaError = math.inf
    def getError(self):
        return self._thetaError
    def dError(self):
        return (self._thetaError - self._prevThetaError)
    def updateError(self, reading):
        self._prevThetaError = self._thetaError
        self._thetaError = get_dTheta(reading, self.thetaTarget)
#### This Target will only drive forward
class MotionTarget(Target):
    def __init__(self, origin_distance, desired_distance:float, ):
        Target.__init__(self, Target.Type.MOTION)
        self.targetDistance = desired_distance
        self.currentDistance = origin_distance
        self.originDistance = origin_distance
        self.is_init = False
    
    def getError(self):
        return self.targetDistance - (self.currentDistance - self.originDistance)
    
    def updateError(self, error:float):
        self.currentDistance = error
class UpdateTarget(Target):
    def __init__(self):
        Target.__init__(self, Target.Type.UPDATE)
        self.is_init = False
    

class Agent:
    MAX_VOLTAGE = 6.28 #V
    WHEEL_DIAMETER = 1.6 #INCHES
    DISTANCE_BETWEEN_WHEELS = 2.28 #INCHES
    ERROR_THRESHOLD_ANGLE = 0.5
    ERROR_THRESHOLD_THETA = 0.1
    ERROR_THRESHOLD_DISTANCE = 0.05
    DERROR_THRESHOLD_DISTANCE = 0.005
    DERROR_THRESHOLD_THETA = 0.05
    
    def __init__(self, use_imu=True, starting_angle=None, is_traversal=False, explore_mode=False):
        #print("Hello World")
        self.robot = Robot()
        self.time_elapsed = 0
        self.time_for_action = 0
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftWheelSensor = self.robot.getDevice('left wheel sensor')
        self.rightWheelSensor = self.robot.getDevice('right wheel sensor')
        self._leftPreviousPosition = 0
        self._rightPreviousPosition = 0
        self._startingLeftPos = 0
        self._startingRightPos = 0
        self._lastKnownPosition = Vector2()
        self.TotalDistanceTraveled = 0
        self.is_traversal = is_traversal
        self.explore_mode = explore_mode
        
        self.ArcRadius = 0
        self.cnt_movements = 0
        self.leftDistanceSensor = self.robot.getDevice('left distance sensor')
        self.rightDistanceSensor = self.robot.getDevice('right distance sensor')
        self.frontDistanceSensor = self.robot.getDevice('front distance sensor')
        self.rearDistanceSensor = self.robot.getDevice('rear distance sensor')
        self.lidar = self.robot.getDevice('lidar')
        lidar_horizontal_res = self.lidar.getHorizontalResolution()
        lidar_num_layers = self.lidar.getNumberOfLayers()
        lidar_min_dist = self.lidar.getMinRange()
        lidar_max_dist = self.lidar.getMaxRange()
        #print(f"Lidar Configuration\nHorizontal Resolution: {lidar_horizontal_res}\nNumber of Layers: {lidar_num_layers}\nMin Range: {lidar_min_dist}\nMax Range: {lidar_max_dist}")
        self.imu = self.robot.getDevice('inertial unit')
        self.cameraFront = self.robot.getDevice('camera1')     
        # self.cameraLeft = self.robot.getDevice('cameraLeft')     
        # self.cameraRight = self.robot.getDevice('cameraRight')     
        # self.cameraRear = self.robot.getDevice('cameraRear')     
        

        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        
        '''Importing Robot Functions for Simplicity'''
        self.getBasicTimeStep = self.robot.getBasicTimeStep
        self.getTime = self.robot.getTime
        
        
        self.timestep = int(self.getBasicTimeStep())
        for device in [self.leftWheelSensor, self.rightWheelSensor, self.leftDistanceSensor, self.rightDistanceSensor, self.frontDistanceSensor, self.rearDistanceSensor, self.imu, self.lidar, self.cameraFront]:#, self.cameraRear, self.cameraLeft, self.cameraRight, ]:
                device.enable(self.timestep) 
        for camera in [self.cameraFront]:#, self.cameraRear, self.cameraLeft, self.cameraRight]:    
            camera.recognitionEnable(self.timestep)   
                
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)   
        
        self.use_imu = use_imu        
        self._last_theta_no_imu = starting_angle
        # self.client = Client(ROBOT_SERVER, authkey=b'EPUCKROBOT')
        self.dest_deq = deque()
        self.curr_dest = None
        
        # self.kinematics_deq = deque()
        
        self._currErrorDist = 0
        self._prevErrorDist = 0
        
        self._currErrorTheta = 0
        self._prevErrorTheta = 0
        
        self.kPDist = 2.4 #0.5
        self.kIDist = 0.1#1.3
        self.kDDist = 0.1
        
        self.kPTheta = 0.7
        self.kITheta = 0.15
        self.kDTheta = 0.2
        
        self.kPObj = 5.0
        self.kIObj = 0.005
        self.kDObj = 0.3
        
        self.m2g = None
        
        self.temperature = 0.9
        self.decay = 0.95
        
        self.grid = None
        self._TransitionModel = None
        self.triangulated = False
        self._Triangulator = LandmarkTriangulator()
        self._Triangulator.addLandmark(-2*Grid.CELL_SIZE, 2*Grid.CELL_SIZE, LandmarkTriangulator.Color.YELLOW)
        self._Triangulator.addLandmark(-2*Grid.CELL_SIZE, -2*Grid.CELL_SIZE, LandmarkTriangulator.Color.GREEN)
        self._Triangulator.addLandmark(2*Grid.CELL_SIZE, 2*Grid.CELL_SIZE, LandmarkTriangulator.Color.RED)
        self._Triangulator.addLandmark(2*Grid.CELL_SIZE, -2*Grid.CELL_SIZE, LandmarkTriangulator.Color.BLUE)
        self.path_deq = deque()

        self._lastTheta = self.imu.getRollPitchYaw()[2] *180/PI % 360
        self.full_range_image = None
        
        self._observations = [{
            'delta_t': 0,
            'left_distance': self.leftDistanceSensor.getValue() * METERS_TO_INCHES,
            'right_distance':  self.rightDistanceSensor.getValue() * METERS_TO_INCHES,
            'front_distance':  self.frontDistanceSensor.getValue() * METERS_TO_INCHES,
            'rear_distance': self.rearDistanceSensor.getValue() * METERS_TO_INCHES,
            'left_wheel_delta': 0,
            'right_wheel_delta': 0,
            'current_pos': Vector2(),
            'previous_pos': Vector2(),
            'linear_velocity': Vector2(),
            'theta': self.imu.getRollPitchYaw()[2]*180/PI % 360,
            'dtheta': 0,
            'omega': 0,
            'arc_radius': 0,
            'current_cell': -1,
            'lidar': np.zeros((360,)),
        }]
        
        self._actions = [{
            'left_voltage': 0,
            'right_voltage': 0,
        }]
        self._CalibrateSensors()     
        
        
        _ = self.getEncoderReadings()
        self.avg_left_depth, self.avg_right_depth = self.calcLocalDepth()
    
    def send_data(self, data, target):
        packet = {'target': target, 'data': data}
        self.client.send(packet)

    def setPIDDist(self, kP, kI, kD):
        self.kPDist = kP if kP is not None else self.kPDist
        self.kIDist = kI if kI is not None else self.kIDist
        self.kDDist = kD if kD is not None else self.kDDist
        
    def setPIDTheta(self, kP, kI, kD):
        self.kPTheta = kP if kP is not None else self.kPTheta
        self.kITheta = kI if kI is not None else self.kITheta
        self.kDTheta = kD if kD is not None else self.kDTheta
        
    # def getTargetWallTurnAngle(self):
    #     if self.curr_dest.current_side == WallTarget.Side.LEFT:
    #         #print("Wall Target Angle", (self['theta'] - 90) % 360)
    #         return (self['theta'] - 90) % 360
    #     #print("Wall Target Angle", (self['theta'] + 90) % 360)
    #     return (self['theta'] + 90) % 360
    
    def calcLocalDepth(self):
        self.full_range_image = self.lidar.getRangeImage()
        dtheta = 0
        try:
            dtheta = int(get_dTheta(self.curr_dest.thetaGuide, self['theta']))
        except:
            pass
        right_side = self.full_range_image[75+dtheta:91+dtheta]
        left_side = self.full_range_image[270+dtheta:286+dtheta]
        return np.average(left_side)*METERS_TO_INCHES, np.average(right_side)*METERS_TO_INCHES
    
    def calcFrontDepth(self):
        self.full_range_image = self.lidar.getRangeImage()
        frontRight = self.full_range_image[::30] #90
        frontLeft = self.full_range_image[360-30::] #270
        return np.average(frontLeft)*METERS_TO_INCHES, np.average(frontRight)
    
    def distToWall(self):
        self.full_range_image = self.lidar.getRangeImage()
        # dtheta = int(get_dTheta(self.curr_dest.thetaGuide, self['theta']))
        frontRight = min(self.full_range_image) #90
        frontLeft = min(self.full_range_image)
        return frontLeft*METERS_TO_INCHES, frontRight*METERS_TO_INCHES

    def distToFront(self):
        self.full_range_image = self.lidar.getRangeImage()
        front = min(min(self.full_range_image[-30::]), min(self.full_range_image[::30]))
        return front*METERS_TO_INCHES
    
    # def lostWall(self):
    #     front_distance = self.full_range_image[0]*METERS_TO_INCHES
    #     self.current_side = self.curr_dest.current_side
    #     self.avg_left_depth, self.avg_right_depth = self.calcLocalDepth()
    #     if self.current_side == WallTarget.Side.LEFT:
    #         return self.avg_left_depth > 14 and front_distance < 15
    #     else:
    #         return self.avg_right_depth > 14 and front_distance < 15
        
    def dErrorDist(self):
        return (self._currErrorDist - self._prevErrorDist)
    def dErrorTheta(self):
        return (self._currErrorTheta - self._prevErrorTheta)
    def resetError(self):
        self._prevErrorTheta = 0
        self._prevErrorDist = 0
        self._currErrorDist = 0
        self._currErrorTheta = 0    

    def _CalibrateSensors(self):
        while self.Status() != -1:
            self.leftMotor.setVelocity(1)
            self.rightMotor.setVelocity(1)
            break
        
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)   
        self._startingLeftPos, self._startingRightPos = self.getEncoderReadings()
        self._leftPreviousPosition, self._rightPreviousPosition = self.getEncoderReadings() 
        # self.step(self.timestep)
    def Status(self):
        return self.robot.step(self.timestep)
    def reset(self):
        pass
    def getIMUReading(self):
        return self['theta'] if self.use_imu else self._last_theta_no_imu
    def getObservations(self):
        return self._observations
    def triangulate(self):
        self.dest_deq.clear()
        self._Triangulator.clearBuffer()
        self.dest_deq.append(TriangulateTarget(self['theta']))
    
    def getEncoderReadings(self):
        return (self.leftWheelSensor.getValue() - self._startingLeftPos, self.rightWheelSensor.getValue()-self._startingRightPos)
    
    def _inAngleRange(self, a, b):
        return abs(abs(a) - abs(b)) < self.ERROR_THRESHOLD_ANGLE
    def _inDistRange(self, a, b):
        return abs(a) - abs(b) < self.ERROR_THRESHOLD_DISTANCE
    
    def _makeObservation(self, delta):
        observation = {
            'delta_t': delta,
            'left_distance': self.leftDistanceSensor.getValue() * METERS_TO_INCHES,
            'right_distance': self.rightDistanceSensor.getValue() * METERS_TO_INCHES,
            'front_distance': self.frontDistanceSensor.getValue() * METERS_TO_INCHES,
            'rear_distance': self.rearDistanceSensor.getValue() * METERS_TO_INCHES,
            'previous_pos': self._lastKnownPosition,
            'theta': self.imu.getRollPitchYaw()[2]*180/PI % 360,
            'lidar': np.array(self.lidar.getRangeImage()) * METERS_TO_INCHES,
            'current_cell': -1,
        }
        front_img = self.cameraFront.getImage()
        observation['cameraFront'] = np.frombuffer(front_img, np.uint8).reshape((self.cameraFront.getHeight(), self.cameraFront.getWidth(), 4))[..., :3].reshape(3, 80, 80)
        # left_img = self.cameraLeft.getImage()
        # observation['cameraLeft'] = np.frombuffer(left_img, np.uint8).reshape((self.cameraLeft.getHeight(), self.cameraLeft.getWidth(), 4))[..., :3].reshape(3, 80, 80)
        # right_img = self.cameraRight.getImage()
        # observation['cameraRight'] = np.frombuffer(right_img, np.uint8).reshape((self.cameraRight.getHeight(), self.cameraRight.getWidth(), 4))[..., :3].reshape(3, 80, 80)
        # rear_img = self.cameraRear.getImage()
        # observation['cameraRear'] = np.frombuffer(rear_img, np.uint8).reshape((self.cameraRear.getHeight(), self.cameraRear.getWidth(), 4))[..., :3].reshape(3, 80, 80)

        lPos, rPos = self.getEncoderReadings()
        omegaLeft = (lPos - self._leftPreviousPosition) / (self.timestep / 1000)
        omegaRight = (rPos - self._rightPreviousPosition) / (self.timestep / 1000)
    
        if self.grid != None and self.grid.current_cell != None:
            observation['current_cell'] = self.grid.current_cell.grid_cell_number()
        # #print(f"Left Wheel: {omegaLeft} \nRight Wheel: {omegaRight}")
        
        observation['omega'] = -(omegaLeft*self.WHEEL_DIAMETER/2 - omegaRight*self.WHEEL_DIAMETER/2) / self.DISTANCE_BETWEEN_WHEELS
        
        
        if not self.use_imu:
            observation['theta'] = (self._last_theta_no_imu + (180/PI * observation['omega'] * self.timestep / 1000)) % (360)
            self._last_theta_no_imu = observation['theta']
        
        observation['left_wheel_velocity'] =  Vector2(r=omegaLeft*self.WHEEL_DIAMETER/2, theta=observation['theta'])
        observation['right_wheel_velocity'] = Vector2(r=omegaRight*self.WHEEL_DIAMETER/2, theta=observation['theta'])
        # #print(f"Left Wheel: {observation['left_wheel_velocity']} \nRight Wheel: {observation['right_wheel_velocity']}")
        try:
            observation['arc_radius'] = (self.DISTANCE_BETWEEN_WHEELS/2) * abs((observation['left_wheel_velocity'].r+observation['right_wheel_velocity'].r) / (observation['left_wheel_velocity'].r - observation['right_wheel_velocity'].r))
        except ZeroDivisionError:
            observation['arc_radius'] = math.inf
                
        observation['linear_velocity'] = (observation['left_wheel_velocity'].plus(observation['right_wheel_velocity'])).scale(0.5)
        
        observation['current_pos'] =  self._lastKnownPosition.plus(observation['linear_velocity'].scale((self.timestep / 1000)))
        self.TotalDistanceTraveled += observation['current_pos'].minus(self._lastKnownPosition).r
        self._lastKnownPosition = observation['current_pos']
        

        # observation['current_vel'] =  observation['current_pos'].minus(observation['previous_pos']).scale(1/delta)
        
        observation['dtheta'] = get_dTheta(self._lastTheta, observation['theta'])
        
        self._lastTheta = observation['theta']
        self._leftPreviousPosition = lPos
        self._rightPreviousPosition = rPos
        
        self._observations.append(observation)
        
    def _makeAction(self):
        action = {
            'left_voltage': self.leftMotor.getVelocity(),
            'right_voltage': self.rightMotor.getVelocity(),
        }
        self._actions.append(action)    
    
    def getObservation(self, index=-1):
        return self._observations[index]
    
    def getAction(self, index=-1):
        return self._actions[index]
    
    def setWheelVoltage(self, left, right):
        voltages = np.array([left, right])
        voltages = np.clip(voltages, -self.MAX_VOLTAGE, self.MAX_VOLTAGE)
        self.leftMotor.setVelocity(voltages[0])
        self.rightMotor.setVelocity(voltages[1])
    
    ''' Agent Step Function for an Enviornment '''
    def step(self, delta):
        global Search_Lock, Search_Thread, Running_Search, Return_Search_Value, Needs_Read_Value
        #print('sdmnsa')
        # if self.time_elapsed == 0:
        #     self.send_data(None, 'start')
        # elif self.time_elapsed > 10000:
        #     self.send_data(None, 'stop')
        # self.time_elapsed += delta
        # self.time_for_action+=delta
        # ''' Step Code Goes Here '''
        self._makeObservation(delta)        
        # self._makeAction()
        lastObs = self.getObservation()
        with Search_Lock:
            if Running_Search:
                return
            elif Needs_Read_Value:
                #print('Reading val')
                self.path_deq = deque(Return_Search_Value[1::])
                Return_Search_Value = []
                #print(self.path_deq)
                #print('plaksjdnsd q')
                #print(self.grid.current_cell)
                Needs_Read_Value = False
                return
        
        # self.send_data(lastObs, 'state')
        ''' PID MOVEMENT '''
        if self.curr_dest != None:
            if self.curr_dest.type == Target.Type.MOTION:
                print('motion')
                
                if not self.curr_dest.is_init:
                    self.curr_dest.originDistance = self.TotalDistanceTraveled
                    self.curr_dest.is_init = True
                self.curr_dest.updateError(self.TotalDistanceTraveled)
                e_t = self.curr_dest.getError()
                v = e_t * self.kPDist
                self.setWheelVoltage(max(max(v, 5.5), 0), max(max(v, 5.5), 0))
                if e_t < 0.05:
                    self.curr_dest = None
                    self.setWheelVoltage(0, 0)
                
                
            elif self.curr_dest.type == Target.Type.TURN:
                '''Update the error from the last observation for each side'''
                # print('Turning...')
                self.curr_dest.updateError(self['theta'])
                print('hergaba')
                e_t = self.curr_dest.getError()
                # dir = dTheta/abs(dTheta)
                u_t = self.kPTheta * e_t  + self.kITheta*e_t *abs(e_t) + self.kDTheta*self.curr_dest.dError()
                # u_t = max(min(u_t, 1.5), 0)
                voltages = np.array([-u_t, u_t])
                voltages = np.clip(voltages, -5., 5.)
                self.setWheelVoltage(voltages[0], voltages[1])
                # print(f"Voltages: {voltages[0]}, {voltages[1]}")
                # print(f'Error: {e_t}')
                if abs(self.curr_dest.dError()) < self.ERROR_THRESHOLD_THETA and abs(e_t) < self.DERROR_THRESHOLD_THETA:
                    self.setWheelVoltage(0, 0)
                    print('Ending Turn.')
                    self.curr_dest = None
                    for _ in range(10):
                        if self.Status() == -1:
                            print("FAILURE")
                            exit(1)
                        self.time_elapsed += delta
                        self.time_for_action+=delta
                        self._makeObservation(delta)
            elif self.curr_dest.type == Target.Type.STOP:
                self.setWheelVoltage(0,0)
                self.curr_dest = None
            elif self.curr_dest.type == Target.Type.DISTANCE:
                print('Distance...')
                reading = self.lidar.getRangeImage()[0]*METERS_TO_INCHES + self.DISTANCE_BETWEEN_WHEELS/4
                err = reading - self.curr_dest.distance
                
                y_t = 2.5 * err
                voltages = np.array([y_t, y_t])
                self.setWheelVoltage(voltages[0], voltages[1])
                if abs(err) < 0.1:
                    self.setWheelVoltage(0, 0)
                    self.curr_dest = None
                    print('Done')
                    print(self.dest_deq)
            elif self.curr_dest.type == Target.Type.UPDATE:
                self.grid.update_barriers(self.lidar.getRangeImage(), self['theta'])
                self.curr_dest = None
                
            # elif self.curr_dest.type == Target.Type.TRI:
            #     lidar_img = self.lidar.getRangeImage()
            #     self.curr_dest.update(self['theta'])
            #     self.setWheelVoltage(-1.5, 1.5)
            #     objs = self.cameraFront.getRecognitionObjects()
            #     fov_cam = self.cameraFront.getFov()
            #     pixels_on_diag = (40**2 + 40**2)**0.5
            #     degrees_per_pixel = fov_cam / pixels_on_diag
            #     for obj in objs:
            #         size = obj.getSize()
            #         dpos_on_img = get_dTheta(obj.getPositionOnImage()[0], 40) #this code does angle stuff with objects
            #         # https://stackoverflow.com/questions/17499409/opencv-calculate-angle-between-camera-and-pixel
            #         offset = degrees(degrees_per_pixel * dpos_on_img)
            #         dpos = Vector2(r=(((obj.getPosition()[0])* METERS_TO_INCHES)**2 + (obj.getPosition()[1]* METERS_TO_INCHES)**2  )**0.5 , theta=lastObs['theta'] + offset).plus(Vector2(r=self.DISTANCE_BETWEEN_WHEELS/2, theta=lastObs['theta'] + offset))
            #         # dpos = Vector2(r=lidar_img[0]*METERS_TO_INCHES , theta=lastObs['theta'] + offset).plus(Vector2(r=self.DISTANCE_BETWEEN_WHEELS/2, theta=lastObs['theta'] + offset))
            #         if abs(offset) < 6:
            #             # dpos = Vector2(r=lidar_img[0] , theta=lastObs['theta'] + offset)#.plus(Vector2(r=self.DISTANCE_BETWEEN_WHEELS/2, theta=lastObs['theta'] + offset))
            #             color = obj.getColors()
            #             color = [color[0], color[1], color[2]]
            #             ident_obj = LandmarkTriangulator.IdentifiedObject(dpos, LandmarkTriangulator.get_color(color))
            #             if not self._Triangulator.hasColor(ident_obj.color):
            #                 self._Triangulator.storeIdentifiedObject(ident_obj)
            #     if self.curr_dest.revolutions() >= 1:
            #         self.setWheelVoltage(0, 0)
            #         print('Ending Triangulation.')
            #         self.curr_dest = None
            #         self.triangulated = True # Temporarily turn off triangulation
            #         if self._Triangulator.canTriangulate():
            #             triangulated_position =  self._Triangulator.Triangulate()
            #             cell_num = Grid.get_cell_from_position(triangulated_position)
            #             if cell_num == None:
            #                 print('Unable to triangulate.')
            #                 return
            #             print(self.grid.grid_generator.current_cell)
            #             self.grid.grid_generator.update_current_pos(Vector2(x=int(cell_num[0]), y=cell_num[1]))
            #             old_grid = self.grid
            #             try:
            #                 self.grid = self.grid.translate_grid_generator()
            #             except:
            #                 self.grid = old_grid
            #                 self.grid.grid_generator.discovered_positions = {}
            #             self.grid.update_barriers_from_memory()
            #             print('Triangulated to:', cell_num[0], cell_num[1])
            #             print(self.grid.print_visited_cells())
            #         else:
            #             print('Unable to triangulate.')
                    
            #     # elif self.curr_dest.revolutions() >= 3: # Sometimes the robot can't triangulate because the objects are unobservable
            #     #     print('Unable to triangulate.')
            #     #     self.setWheelVoltage(0, 0)
            #     #     print('Using Last Known Position.')
            #     #     self.curr_dest = None
            #     #     self.triangulated = True
                
            if self.curr_dest is not None:
                self.curr_dest.action_time+=delta 
                
        elif len(self.dest_deq) > 0:
            self.curr_dest = self.dest_deq.popleft()
            self.curr_dest.action_time = 0
            #print(self.dest_deq)
            
        else:
            self.grid.update_barriers(self.lidar.getRangeImage(), self['theta'])
            if self.grid.location_known() and self.grid.UseGridGenerator:
                        #print('Updating Path...')
                        #print(self.grid.grid_generator.discovered_positions)
                        self.grid = self.grid.translate_grid_generator()
                        self.grid.update_barriers_from_memory()
            try:
                with open('out.txt', 'w') as f:
                    f.write(grid.__str__())
            except:
                pass
            # #print(self.grid)
            try:
                print(f'Agent State: {self.grid.get_current_grid_cell_number()}, Pos({self["current_pos"].x}, {self["current_pos"].y}) Theta {self["theta"]}')
            except KeyError:
                print(f'Agent State: UNKOWN CELL, UKNOWN POS, Theta {self["theta"]}')
            if len(self.path_deq) > 0:
                next_cell = self.path_deq.popleft()
                action = None
                for dir, edge in self.grid.get_current_cell().edges.items():
                    # #print(dir, edge)
                    if edge and edge.to == next_cell:
                        action = dir
                        break
                origin_node = self.grid.get_current_cell()
                if self._TransitionModel.TransitionState(action, self.path_deq):
                    # if np.random.sample() < self.grid.follow_prob(self.temperature):
                        # self.path_deq.clear()
                        # self.temperature *= self.decay
                    return
                else:
                    print('Gjnfdksjnfk ')
                    self.path_deq.clear()
                    self.grid.set_current_cell(origin_node.grid_cell_number())
                    return
            while True:
                
                try:
                    if False and self.is_traversal:
                        self.path_deq = deque(self.grid.find_path_to(self.grid.destination_cell))
                        if self.grid.current_cell == self.grid.destination_cell:
                            #print('Acomplished Shortest Path...')
                            exit(0)
                    else:
                        # path = self.grid.rec_search(self.grid.current_cell, self.grid.destination_cell, max_frontier_size=10, stop_depth=20)# self.grid.get_optimal_unvisited_exploration(Grid.Node.Edge.Type.getDirection(get_closestCardinalDirection(self['theta'])))
                        # self.path_deq = deque(self.grid.find_path_to(self.grid.get_last_discovered()))
                        # if len(self.path_deq) == 0:
                        print('fdnhkjsadfkjhbas')
                        with Search_Lock:
                            Search_Thread =  thr.Thread(target=self.grid.call_rec_unvisited_search, args=(self.grid.current_cell, self.grid.destination_cell), kwargs={'stop_depth':300, 'max_frontier_size':-1})
                            Return_Search_Value = []
                            Running_Search = True
                            Needs_Read_Value = False
                            Search_Thread.start()
                        
                        # #print(path)
                        # if isinstance(path, Grid.Path):
                        #     #print('aaaaa')
                        #     self.path_deq = deque(path.path)
                        # elif isinstance(path, Grid.Node):
                        #     #print('bluwho')
                        #     self.path_deq = deque([self.grid.current_cell, path])
                        # else:
                        #     #print('FATALITI')  
                        
                except (ValueError, IndexError) as e: # note
                    #print(e)
                    #print('Acomplished Shortest Path...')
                    Grid.save_to_file(self.grid, 'GridState.json')
                    exit(0)
                    
           
                break
            
        
    def __getitem__(self, key):
        return self.getObservation()[key]
    
    def __str__(self):
        obs = self.getObservation()
        return f"Time elapsed: {self.time_elapsed} \n" \
            f"Position: {obs['current_pos']} \n" \
                f"Angular Velocity: {obs['omega']} \n" \
                    f"Theta: {obs['theta']} \n"\
                        f"Linear Velocity: {obs['linear_velocity']} \n" \
                        f"Distance Traveled: {self.TotalDistanceTraveled} \n" \
                        f"Arc Radius: {obs['arc_radius']}\n"
                            
    
    ### Getters to make things Easier ####                
    def set_pos(self, pos):
        self._lastKnownPosition = Vector2(x=pos[0], y=pos[1])
        self._last_theta_no_imu = pos[2]
    
    def get_pos(self):
        return self._lastKnownPosition
    def get_theta(self,):
        return self['theta']
    def set_grid(self, grid):
        self.grid = grid
        self._TransitionModel = TransitionModel(self, self.grid)
    
# 
# import torch as th

# address = ('localhost', 31313)
# conn = 
# conn.send('Hello World')
# data = {
#     'req': 'test', 'state': 0, 'action': np.array([1, 2, 1]), 'delta_t': 0.02,
# }
# conn.send(th.tensor([1, 2, 3]))
# conn.send(data)
# # conn.send('close')
# # can also send arbitrary objects:
# # conn.send(['a', 2.5, None, int, sum])
# conn.close()      
        
        
'''COPY ENDS HERE''' 
# while True:
#     time.sleep(1)
    
import os
# create the Robot instance.
agent = Agent(use_imu=True)
if os.path.exists('GridState.json'):
    grid = Grid.load_from_file('GridState.json')
else:
    grid = Grid(use_grid_generator=False)
grid.set_current_cell_pos(15, 0)
grid.set_destination_cell(120)
# #print(grid)
agent.set_grid(grid)
#print(agent.grid)
i = 0
start = time.time()
while agent.Status() != -1:
        agent.step(agent.timestep)
        
        if time.time() - start > 60*10:
            Grid.save_to_file(agent.grid, 'GridState.json')
            exit(0)
            
        # i+=1
        # if i == 2000:
        #     break
