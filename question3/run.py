from main import EdgeWeightedGraph,Edge

file = open('usa.txt')
V,E = ( int(i) for i in file.readline().split())

print(V,E)

graph = EdgeWeightedGraph(V)

for line in file:
    v,w,weight = (int(i) for i in line.split())
    graph.addEdge(Edge(v,w,weight))
a=graph.min_path(1,2)
#a=graph.dijkstra(1)
#a=graph.dijkstra_heap(1)
#graph.dijkstra(1)
#ng = graph.toNxGraph()
