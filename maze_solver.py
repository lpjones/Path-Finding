import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class Node(object):
    def __init__(self, x, y, distance=float('inf'), prev_node=None, visited=False):
        self.x = x # Rows
        self.y = y # Columns
        self.distance = distance
        self.visited = visited # 1 if true
        self.edges = [] # Neighbors of Node
        self.prev_node = prev_node # Previous node in the current best path


class Graph(object):
    def __init__(self):
        self.nodes = []

    def get_node(self, x, y): # Get a specific node
        for node in self.nodes: # Go through all nodes currently in graph
            if node.x == x and node.y == y:
                return node # Return specific node
        print("node not found")
    
    def append_node(self, x, y, distance, visited):
        for node in self.nodes: # Check to see if node already exists in graph
            if node.x == x and node.y == y:
                return
        # If node not already in graph, append
        new_node = Node(x, y, distance, visited)
        self.nodes.append(new_node)


    def append_edge(self, node_1, node_2):
        node_1.edges.append(node_2)
        node_2.edges.append(node_1)
        
    
class PriorityQueue(object):
    def __init__(self):
        self.queue = []
 
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
 
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
 
    # for inserting an element in the queue
    def push(self, data):
        self.queue.append(data)
 
    # for popping an element based on Priority
    def pop(self):
        try:
            min_node = 0
            for i in range(len(self.queue)):
                if self.queue[i].distance < self.queue[min_node].distance:
                    min_node = i
            item = self.queue[min_node]
            del self.queue[min_node]
            return item
        except IndexError:
            print()
            exit()

def read_maze(file_path):
    """
    Read the maze from a text file and convert it into a 2D numpy array.
    
    Parameters:
    - file_path: str, the path to the maze text file.
    
    Returns:
    - maze: 2D numpy array, representation of the maze where ' ' indicates a path, '#' indicates a wall,
            's' is the start, and 'e' is the end.
    """
    # Check if input maze txt file exists
    if not os.path.isfile(file_path):
        print(f"{file_path} is not a valid file path")
        exit(0)

    # Read in the file
    with open(file_path, 'r') as file:
        lines = file.readlines() # Read in the lines of the file

    lines = [line.strip() for line in lines] # Strip the whitespace from each line

    maze = np.array([list(line) for line in lines]) # Create 2D numpy array

    return maze

def binary_maze(maze_mat):
    bin_mat = np.zeros(maze_mat.shape)
    start, end = (0, 0), (0, 0)
    for r, i in enumerate(maze_mat):
        for c, j in enumerate(i):
            if j == '#':
                bin_mat[r][c] = 1
            elif j == 's':
                start = (r, c)
            elif j == 'e':
                end = (r, c)
    return bin_mat, start, end


def a_star_algorithm(maze):
    """
    Implement the A* algorithm to find the shortest path through the maze from start to end.
    
    Parameters:
    - maze: 2D numpy array, the maze to navigate.
    
    Add more parameters if you see fit
    """
    pass  # Your code here

def mat_to_graph(bin_mat):
    graph = Graph()
    for r, row in enumerate(bin_mat): # Loop through each row of binary matrix
        for c, val in enumerate(row): # Looko at value in bin_mat[][]
            if val == 0: 
                graph.append_node(r, c, distance=float('inf'), visited=False)  # Append the node
                curr_node = graph.get_node(r, c) # Get the node you just appended
                if bin_mat[r + 1][c] == 0: # Look to below of curr_node for neighbors
                    graph.append_node(r + 1, c, distance=float('inf'), visited=False)
                    graph.append_edge(curr_node, graph.get_node(r + 1, c)) # Add edge between curr_nod and neighbor node
                if bin_mat[r][c + 1] == 0: # Look to the right of curr_node for neighbors
                    graph.append_node(r, c + 1, distance=float('inf'), visited=False)
                    graph.append_edge(curr_node, graph.get_node(r, c + 1))
            
    return graph


def dijkstras_algorithm(graph, start_node):
    """
    Implement Dijkstra's algorithm to find the shortest path through the maze from start to end.
    
    Parameters:
    - maze: 2D numpy array, the maze to navigate.
    
    Add more parameters if you see fit
    """
    pq = PriorityQueue()

    start_n = graph.get_node(start_node[0], start_node[1]) # Get x and y of start_node
    start_n.distance = 0
    start_n.prev_node = start_n
    pq.push(start_n) 

    while len(pq.queue) > 0:
        curr_node = pq.pop()
        curr_node.visited = True

        for neighbor in curr_node.edges:

            # Only consider this new path if it's better than any path we've
            # already found.
            if curr_node.distance + 1 < neighbor.distance:
                neighbor.distance = curr_node.distance + 1
                neighbor.prev_node = curr_node # Update the best path
                if neighbor.visited == False:
                    pq.push(neighbor)

    return graph


def plot_maze(maze, visited, current):
    """
    Visualize the maze, the visited cells, the current position, and the end position.
    
    Parameters:
    - maze: 2D numpy array, the maze to visualize.
    - visited: visited coordinates.
    - current: current position in the maze.

    Add more parameters if you see fit
    """
    pass  # Your code here



"""
Parse command-line arguments and solve the maze using the selected algorithm.
"""

def check_input_paths(args):
    
    # Check for correct num of arguments
    if len(args.input_files) < 1:
        print("Invalid number of input files. Need 1") 
        exit(0)

    if args.algorithm != 'astar' and args.algorithm != 'dijkstra':
        print("Invalid algorithm, select 'dijkstra' or 'astar'")
        exit(0)

""" 
Requirements: 
1. Path to Maze txt file
2. Option to select which algorithm to run
3. Flag for visualization (visualization of the searching process, not just the final solved maze)

python maze_solver.py <path_to_maze_txt> --algorithm <dijkstra/astar>
"""

# Parse in arguments from command line
# Referenced from argparse library
parser = argparse.ArgumentParser(description = "Command-Line Parser")

# Define a positional argument, our input file
parser.add_argument('input_files', nargs = 1, type = str, help = '<path_to_maze_txt>')
parser.add_argument('--algorithm', help = "<dijkstra/astar>")
parser.add_argument('-vis', action = "store_true", help = "Visualize Search Process")

args = parser.parse_args()

check_input_paths(args) # Check args for invalid inputs

selected_algo = args.algorithm # Either dijkstra or astar
path_name = args.input_files[0] 

maze_mat = read_maze(path_name)

bin_mat, start_pos, end_pos = binary_maze(maze_mat)

graph = mat_to_graph(bin_mat)

com_graph = None # Complete Graph

if selected_algo == 'dijkstra':
    com_graph = dijkstras_algorithm(graph, start_pos)
elif selected_algo == 'astar':
    pass
    # TODO: 
    
for node in com_graph.nodes:
    print('(', node.x, ',', node.y, ') ', '->', 
          ' (', node.prev_node.x, ',', node.prev_node.y, ')', sep='')


