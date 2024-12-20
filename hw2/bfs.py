import csv
edgeFile = 'edges.csv'

def bfs(start, end):
    # Begin your code (Part 1)
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


    # Perform BFS to find the shortest path
    # initialize a dictionary 'parent' to save the parent node and a list 'queue'. 
    # initialize a set to save the visited node.
    # set num_visited and done = 0.
    parent = {}
    visited = set()
    queue = [start]
    # queue = deque(start)
    visited.add(start)
    # queue = deque([(start, [start], 0)])  # current node, path, distance of path
    num_visited = 0
    done = 0


    while queue:
        # pop the first node in queue
        node = queue.pop(0)
        # node = queue.popleft()
        
        # check if reach the end
        if node == end:
            done = 1
            break
        # if not add 1 to the num_visited
        num_visited += 1
        # record the parent node into visited
        for i, j in edge[node]:
            if i not in visited:
                queue.append(i)
                visited.add(i)
                parent[i] = (node, j)

                
    # find the path and calculate the distance
    # set a list 'path' to save parent node.
    # set dist = 0
    path = []
    dist = 0
    if done:
        path.append(end)
        # calculate the distance until the 'start'
        while path[-1] != start:
            # add the distance from parent
            dist += parent[path[-1]][1]
            path.append(parent[path[-1]][0])
        path.reverse()
    return path, round(dist, 3), num_visited

    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
