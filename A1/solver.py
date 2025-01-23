import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        if node not in self.neighbors:
            self.neighbors.append(node)
        if self not in node.neighbors:
            node.neighbors.append(self)
        pass

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes_dict[(r,c)] = Node((r,c))
    # 2) Link each node with valid neighbors in four directions (undirected)
    for (r,c), node in nodes_dict.items():
        for dr, dc in [(-1, 0), (1,0), (0,-1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in nodes_dict:
                node.add_neighbor(nodes_dict[(nr, nc)])
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    start_node = nodes_dict.get((0,0), None)
    goal_node = nodes_dict.get((rows-1, cols-1), None)

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    if start_node is None or goal_node is None:
        return None
    
    visited = set([start_node])
    queue = deque([start_node])
    parent_map = {start_node: None}

    while queue:
        node = queue.popleft()
        if node == goal_node:
            return reconstruct_path(node, parent_map)
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent_map[neighbor] = node
    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    if start_node is None or goal_node is None:
        return None
    
    visited = set([start_node])
    stack = [start_node]
    parent_map = {start_node: None}

    while stack:
        node = stack.pop()
        if node == goal_node:
            return reconstruct_path(node, parent_map)

        visited.add(node)
        for neighbor in node.neighbors:
            if neighbor not in visited:
                parent_map[neighbor] = node
                stack.append(neighbor)
    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    if start_node is None or goal_node is None:
        return None
    
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    visited = {}

    g_score = {start_node: 0}
    f_score = {start_node: manhattan_distance(start_node, goal_node)}

    
    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            return reconstruct_path(current, visited)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            
            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                visited[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal_node)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    r1, c1 = node_a.value
    r2, c2 = node_b.value

    return abs(r1 - r2) + abs(c1 - c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    if start_node is None and goal_node is None:
        return None

    start_frontier = deque([start_node])
    goal_frontier = deque([goal_node])

    visited_start = {start_node: None}
    visited_goal = {goal_node: None}

    while start_frontier and goal_frontier:
        if start_frontier:
            current_start = start_frontier.popleft()
            for neighbor in current_start.neighbors:
                if neighbor not in visited_start:
                    visited_start[neighbor] = current_start
                    start_frontier.append(neighbor)
                    if neighbor in visited_goal:
                        return reconstruct_bidrectional_path(neighbor, visited_start, visited_goal)
        
        if goal_frontier:
            current_goal = goal_frontier.popleft()
            for neighbor in current_goal.neighbors:
                if neighbor not in visited_goal:
                    visited_goal[neighbor] = current_goal
                    goal_frontier.append(neighbor)
                    if neighbor in visited_start:
                        return reconstruct_bidrectional_path(neighbor, visited_start, visited_goal)
                    
    return None
#Helper: Reconstruct Bidirectional Path

def reconstruct_bidrectional_path(meeting_node, visited_start, visited_goal):
    path_start = []
    path_goal = []

    current = meeting_node
    while current is not None: 
        path_start.append(current.value)
        current = visited_start[current] 
    path_start[::-1]

    current = visited_goal[meeting_node]
    while current is not None:
        path_goal.append(current.value)
        current = visited_goal[current]
    
    return path_start + path_goal


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    if start_node is None and goal_node is None:
        return None
    
    current = start_node
    current_cost = manhattan_distance(current, goal_node)
    path = [current.value]

    while temperature > min_temperature:
        if current == goal_node:
            return path
        
        neighbor = random.choice(current.neighbors)
        next_cost = manhattan_distance(neighbor, goal_node)
        
        cost_diff = next_cost - current_cost

        if next_cost < current_cost:
            current = neighbor
            current_cost = next_cost
            path.append(current.value)
        elif random.random() < math.exp(-cost_diff / temperature):
            current = neighbor
            current_cost = next_cost
            path.append(current.value)
        
        temperature *= cooling_rate

    return None if current != goal_node else path


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    path = []
    current = end_node

    while current is not None:
        path.append(current.value)
        current = parent_map.get(current)
    return path[::-1]


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
