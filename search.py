# Prof. Patrick Winston version
distances = { 'a': 3, 'b': 3, 'c': 4, 'd': 1, 'e': 1, 'f': 4, 'g': 0 }
adjacency_list = { 'a': ['b', 'd'], 'b': ['a', 'c'], 'c': ['b', 'e'], 'd': ['a', 'g'], 'e': ['c'], 'f': ['b', 'a'], 'g': ['d']  }
adjacency_matrix = [
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0]
]

def depth_first_search(start, dest):
    visited = [start]
    queue = [[start]]
    while len(queue) > 0:
        print (queue)
        path = queue.pop()
        node = path[-1]
        for connectedNode in adjacency_list[node]:
            if not connectedNode in visited:
                new_path = list(path)
                new_path.append(connectedNode)
                if connectedNode == dest:
                    return new_path
                visited.append(connectedNode)
                queue.append(new_path)

def breadth_first_search(start, dest):
    visited = [start]
    queue = [[start]]
    while len(queue) > 0:
        print (queue)
        path = queue.pop(0)
        node = path[-1]
        for connectedNode in adjacency_list[node]:
            if not connectedNode in visited:
                new_path = list(path)
                new_path.append(connectedNode)
                if connectedNode == dest:
                    return new_path
                visited.append(connectedNode)
                queue.append(new_path)

def get_distance(node):
    return distances[node]

def hill_climbing_search(start, dest):
    visited = [start]
    queue = [[start]]
    while len(queue) > 0:
        print (queue)
        path = queue.pop()
        node = path[-1]
        not_visited_nodes = [x for x in adjacency_list[node] if x not in visited]
        not_visited_nodes.sort(key=get_distance, reverse=True) # Sort so the greatest distance comes first
        for connectedNode in not_visited_nodes:
            if not connectedNode in visited:
                new_path = list(path)
                new_path.append(connectedNode)
                if connectedNode == dest:
                    return new_path
                visited.append(connectedNode)
                queue.append(new_path)

print (depth_first_search('f', 'g'))
print (breadth_first_search('f', 'g'))
print (hill_climbing_search('f', 'g'))