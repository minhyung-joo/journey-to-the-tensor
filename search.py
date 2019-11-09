# Prof. Patrick Winston version
distances = {'a': 3, 'b': 3, 'c': 4, 'd': 1, 'e': 1, 'f': 4, 'g': 0}
index_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
heuristics = {'a': 7, 'b': 6, 'c': 7, 'd': 5, 'e': 2, 'f': 11, 'g': 0}
adjacency_list = {'a': ['b', 'd'], 'b': ['a', 'c'], 'c': [
    'b', 'e'], 'd': ['a', 'g'], 'e': ['c'], 'f': ['b', 'a'], 'g': ['d']}
distance_matrix = [
    [0, 4, 0, 3, 0, 3, 0],
    [4, 0, 4, 0, 0, 5, 0],
    [0, 4, 0, 0, 6, 0, 0],
    [3, 0, 0, 0, 0, 0, 5],
    [0, 0, 6, 0, 0, 0, 0],
    [3, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0]
]


def depth_first_search(start, dest):
    visited = [start]
    queue = [[start]]
    while len(queue) > 0:
        print(queue)
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
        print(queue)
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
        print(queue)
        path = queue.pop()
        node = path[-1]
        not_visited_nodes = [
            x for x in adjacency_list[node] if x not in visited]
        # Sort so the greatest distance comes first
        not_visited_nodes.sort(key=get_distance, reverse=True)
        for connectedNode in not_visited_nodes:
            if not connectedNode in visited:
                new_path = list(path)
                new_path.append(connectedNode)
                if connectedNode == dest:
                    return new_path
                visited.append(connectedNode)
                queue.append(new_path)

# Already assumes the destination is g


def a_star_search(start):
    visited = []
    queue = [[start]]
    distances_queue = [0]
    scores_queue = [10]
    found_distance = float("inf")
    found_path = None
    while len(queue) > 0:
        # Dequeue path with least score
        # score = traversed_distance + heuristic
        minimum_score_index = scores_queue.index(min(scores_queue))
        scores_queue.pop(minimum_score_index)
        path = queue.pop(minimum_score_index)
        current_distance = distances_queue.pop(minimum_score_index)
        if current_distance < found_distance:
            node = path[-1]
            visited.append(node)
            new_nodes = [x for x in adjacency_list[node] if not x in visited]
            for new_node in new_nodes:
                new_path = list(path)
                new_path.append(new_node)
                new_distance = current_distance + \
                    distance_matrix[index_map[node]][index_map[new_node]]
                print(new_path)
                print(new_distance)
                if not new_node == 'g':
                    queue.append(new_path)
                    distances_queue.append(new_distance)
                    scores_queue.append(new_distance + heuristics[new_node])
                else:
                    found_path = new_path
                    found_distance = new_distance
    print(found_distance)
    return found_path


print(a_star_search('f'))
