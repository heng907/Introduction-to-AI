import csv
import heapq
edgeFile = 'edges.csv'

def ucs(start, end):
    # Begin your code (Part 3)
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


    # set a 'parent' dictionary and a heapfied list 'heap'
    # set num_visited, done, dist = 0 
    parent = {}
    heap = [(0, start, None)]
    heapq.heapify(heap)
    visited = set()
    num_visited = 0
    done = 0
    dist = 0


    while heap:
        # pop the first 
        (cost, node, p) = heapq.heappop(heap)
        # check if reach the 'end'
        if node == end:
            done, dist = 1, cost
            parent[node] = p
            break
        # If the current node is unvisited, add the node to the 'visited'.
        # Then add 1 to the 'num_visited'.
        if node not in visited:
            num_visited += 1
            visited.add(node)
            parent[node] = p
            for i, j in edge[node]:
                heapq.heappush(heap, (cost + j, i, node))

    path = [] # initialize a list of path
    if done == 1 :
        path.append(end)
        while path[-1] != start:  # to read the first and check if it is start
            path.append(parent[path[-1]]) # add the parent id to the path
        path.reverse() # finally reverse the path
    return path, round(dist, 3), num_visited


#     # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
