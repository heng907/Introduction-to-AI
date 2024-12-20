import csv
import queue
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    # raise NotImplementedError("To be implemented")


    # Load the graph from the edges.csv file
    edge = {}
    with open(edgeFile, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start_node = int(row['start']) # read the start
            end_node = int(row['end']) # read the end
            distance = float(row['distance']) # read the distance

            if start_node not in edge:
                edge[start_node] = [(end_node, distance)]
            else:
                edge[start_node].append((end_node, distance))
            if end_node not in edge:
                edge[end_node] = []
    # Implement DFS using a stack instead of a queue
    parent = {}
    stack = [start]  # Use a stack to store the nodes to visit
    visited = set()
    visited.add(start)
    num_visited = 0
    done = 0
    while stack:
        node = stack.pop()  # Pop from the stack to do DFS
        num_visited += 1
        if node == end:
            done = 1
            break
        for i, j in edge.get(node, []):  # Get the children of the current node, if any
            if i not in visited:
                stack.append(i)  # Push the node to the stack for DFS
                visited.add(i)
                parent[i] = (node, j)  # Store the parent and distance for path reconstruction

    # Find the path and calculate the distance
    path = []
    dist = 0
    if done:  # If the destination has been found
        current_node = end
        while current_node != start:
            path.append(current_node)
            parent_node, parent_distance = parent[current_node]
            dist += parent_distance
            current_node = parent_node
        path.append(start)  # Append the start node at the end
        path.reverse()  # Reverse the path to start from the beginning

    return path, round(dist, 3), num_visited

    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
