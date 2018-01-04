import networkx as nx
import numpy as np
import heapq


class Edge():
    def __init__(self, v = None, w = None, weight = 1):
        self.weight = weight
        self.v = v
        self.w = w
    def either(self):
        return self.v,self.w
    def other(self,v):
        if v==self.v:
            return self.w
        elif v==self.w:
            return self.v
        else:
            raise ValueError("V:{} not in this edge.",v)
    def __str__(self):
        return '{} <-- {} --> {}'.format(self.v,self.weight,self.w)
    def __lt__(self,edge):
        return self.weight < edge.weight
    def __eq__(self,edge):
        return self.weight == edge.weight        
    def __gt__(self,edge):
        return self.weight > edge.weight 


class EdgeWeightedGraph():
    def __init__(self,V):
        self.V = V
        self.E = 0
        self.adj = []
        for i in range(V):
            self.adj.append([])
    def addEdge(self,edge):
        v,w = edge.either()
        self.adj[v].append(edge)
        self.adj[w].append(edge)
        self.E += 1
    def edges(self):
        for i in range(self.V):
            for edge in self.adj[i]:
                if edge.other(i)>i:
                    yield edge
    def toNxGraph(self):
        g = nx.Graph() 
        for edge in self.edges():
            g.add_edge(*edge.either(),weight=edge.weight)   
        return g  
    def draw(self):
        nx.draw(self.toNxGraph())
    def min_path(self,v,w):
        prev = self._dijkstra_heap_end(w,v)
        if prev[v]==-1:
            return None
        yield v
        while not v == prev[v]: 
            v = prev[v]
            yield v
    def _dijkstra_heap_end(self,v = 0,endv = None):
        if not endv:
            return dijkstra_heap(v)
        S = np.ones(self.V) * False
        dist = []
        heapq.heappush(dist,(0,v))
        prev = np.ones(self.V,'int32') * -1
        prev[v] = v
        S[v] = True
        for edge in self.adj[v]:# init
            w = edge.other(v)
            heapq.heappush(dist,(edge.weight, w))
            prev[w] = v
        for i in range(1,self.V):
            if(i%1000==0):
                print('runing {}/{}   num of dist:{}'.format(i,self.V,len(dist)))
            if not dist:
                break
            u = v
            minv = heapq.heappop(dist)
            while S[minv[1]]:
                try:
                    minv = heapq.heappop(dist)
                except IndexError:
                    return prev
            u = minv[1]
            if u == endv:
                return prev            
            S[u] = True
            for edge in self.adj[u]:# change dist
                w = edge.other(u)
                e = False
                for i,(weight,distv) in enumerate(dist):
                    if w == distv:
                        e = True
                        if weight > edge.weight:
                            del dist[i]
                            heapq.heappush(dist,(edge.weight, w))
                            prev[w] = u  
                        break
                if not S[w] and not e:
                    heapq.heappush(dist,(edge.weight, w))
                    prev[w] = u 
        return prev

    def dijkstra_heap(self,v = 0):
        S = np.ones(self.V) * False
        dist = []
        heapq.heappush(dist,(0,v))
        prev = np.ones(self.V,'int32') * -1
        prev[v] = v
        S[v] = True
        for edge in self.adj[v]:# init
            w = edge.other(v)
            heapq.heappush(dist,(edge.weight, w))
            prev[w] = v
        for i in range(1,self.V):
            if(i%1000==0):
                print('runing {}/{}   num of dist:{}'.format(i,self.V,len(dist)))
            if not dist:
                break
            u = v
            minv = heapq.heappop(dist)
            while S[minv[1]]:
                try:
                    minv = heapq.heappop(dist)
                except IndexError:
                    return prev
            u = minv[1]
            S[u] = True
            for edge in self.adj[u]:# change dist
                w = edge.other(u)
                e = False
                for i,(weight,distv) in enumerate(dist):
                    if w == distv:
                        e = True
                        if weight > edge.weight:
                            del dist[i]
                            heapq.heappush(dist,(edge.weight, w))
                            prev[w] = u  
                        break
                if not S[w] and not e:
                    heapq.heappush(dist,(edge.weight, w))
                    prev[w] = u 
        return prev

    def dijkstra(self,v=0):
        S = np.ones(self.V) * False
        dist = np.ones(self.V) * np.inf
        prev = np.ones(self.V,'int32') * -1   
        dist[v] = 0 
        prev[v] = v
        S[v] = True
        for edge in self.adj[v]:# init
            w = edge.other(v)
            dist[w] = edge.weight
            prev[w] = v
        for i in range(1,self.V):
            if(i%100==0):
                print('runing {}/{}'.format(i,self.V))
            mindist = np.inf
            u = v
            for j in range(1,self.V):# find the next V for min dist
                if not S[j] and dist[j] < mindist:
                    u = j
                    mindist = dist[j]
            S[u] = True
            for edge in self.adj[u]:# change dist
                w = edge.other(u)
                if dist[w] > edge.weight:
                    dist[w] = edge.weight
                    prev[w] = u  
        return prev

    def __str__(self):
        return 'Graph:  V:{}  E:{}'.format(self.V,self.E)


if __name__=='__main__':
    a = EdgeWeightedGraph(6)
    a.addEdge(Edge(0,1))
    a.addEdge(Edge(0,3))
    a.addEdge(Edge(1,2))
    a.addEdge(Edge(1,4))
    a.addEdge(Edge(2,4))
    a.addEdge(Edge(2,3))
    a.addEdge(Edge(2,5))
    a.addEdge(Edge(3,5))
    a.addEdge(Edge(4,5))
    
    print(a)
    print(a.dijkstra())
    print(a.dijkstra_heap())
