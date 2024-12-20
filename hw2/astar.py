import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    
    # Load the graph from the edges.csv file
    edge = {}
    with open(edgeFile, 'r') as e_file:
        reader = csv.DictReader(e_file)
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
    
    # Load the graph from the heuristic.csv file
    heur = {}
    with open(heuristicFile) as h_file:
        reader = list(csv.reader(h_file))
        idx = next((i for i in range(1, 4) if reader[0][i] == str(end)))
        # pop the 'title' row
        reader.pop(0)
        # record the data into the 'heur' dictionary
        for row in reader:
            heur[int(row[0])] = float(row[idx])


    # implement the a*
    # initial a 'parent' dictionary and a priority queue 'queue'
    parent = {}
    heap = [(heur[start], start, None)]
    heapq.heapify(heap)
    # set a set 'visited' to save the visited node
    #set num_visited, done, dist = 0
    visited = set()
    num_visited = 0
    done = 0
    dist = 0


    while heap:
        # pop the first tuple in the priority queue 'queue'
        (cost, node, current) = heapq.heappop(heap)
        cost -= heur[node]

        if node not in visited:
            # if not add 1 to the num_visited
            num_visited += 1
            visited.add(node)
            parent[node] = current
            if node == end:
                done, dist = 1, cost
                break
            # explore the neighbor node
            for i, j in edge[node]:
                heapq.heappush(heap, (cost + j + heur[i], i, node))

    # Initial a list 'path'
    path = []
    if done:
        path.append(end)
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
    return path, round(dist, 3), num_visited

    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
