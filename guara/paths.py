
import numpy as np
from guara.graph import Graph
from guara.utils import weighted_heap

def breadth_search_mtx(graph, seed=0):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # parent[v]: o indice do vertice pai de v na arvore gerada pela busca
    #
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1

    # [NOTE] Se um vertice terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    distance[seed] = 0 # definimos o nivel da raiz como 0
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = [seed]

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue.pop(0) # retiramos ele da fila

        for vert in range(len(graph)): # iremos percorrer todos os vertices para ver se sao vizinhos do atual
            if graph[current][vert]: # existe uma aresta (current,vert), isto eh, eles sao vizinhos
                if vert != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                    if not visited[vert]: # vert ainda nao foi investigado
                        queue.append(vert) # adicionamos vert a fila para ser investigado
                        visited[vert] = True # indicamos que vert ja foi descoberto
                        
                        distance[vert] = distance[current] + 1 # declaramos vert como estando 1 nivel acima de current
                        parent[vert] = current # declaramos current como o vertice pai de vert na arvore

    return (distance, parent)

# -----------------------------------------------


def breadth_search_lst(graph, seed=0):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # parent[v]: o indice do vertice pai de v na arvore gerada pela busca
    #
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1

    distance[seed] = 0 # definimos o nivel da raiz como 0
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = [seed]

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue.pop(0) # retiramos ele da fila

        for neighbor in graph[current]: # verificamos todos os vizinhos do vertice sendo investigado
            if neighbor != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[neighbor]: # o vizinho ainda nao foi investigado
                    queue.append(neighbor) # adicionamos o vizinho a lista para ser investigado
                    visited[neighbor] = True # indicamos que o vizinho ja foi descoberto
        
                    distance[neighbor] = distance[current] + 1 # declaramos neighbor como estando 1 nivel acima de cur
                    parent[neighbor] = current # declaramos cur como o vertice pai de neighbor na arvore

    return (distance, parent)

# -----------------------------------------------


def depth_search_mtx(graph, seed=0):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(graph.n, dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # parent[v]: o indice do vertice pai de v na arvore gerada pela busca
    #
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1

    # [NOTE] Se um vertice terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    distance[seed] = 0 # definimos o nivel da raiz como 0
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # stack: pilha com os indices dos vertices que precisam ser investigados
    #   |_ stack[-1]: o vertice a ser investigado no momento
    #
    stack = [seed]

    while (len(stack) != 0):
        current = stack[-1] # pegamos o indice do vertice sendo investigado
        stack.pop(-1) # retiramos ele da pilha
        
        if not visited[current]:
            visited[current] = True

            for vert in range(len(graph))[::-1]: # pegamos todos os elementos da lista usando step=-1
                # seguindo o padrao da aula 5 (slide 18) percorremos os vertices em ordem descrescente
                # desse modo os vizinhos de menor indice ficam sempre no topo da pilha, isto eh, sao analisados primeiro
                if graph[current][vert]: # se os dois forem vizinhos
                    if not visited[vert]: # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                        stack.append(vert) # adicionamos o vizinho na pilha

                        distance[vert] = distance[current] + 1 # o nivel do noh vert eh um nivel acima do atual
                        parent[vert] = current # o pai do noh vert eh o no sendo analisado

    return (distance, parent)

# -----------------------------------------------


def depth_search_lst(graph, seed=0):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(graph.n, dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # parent[v]: o indice do vertice pai de v na arvore gerada pela busca
    #
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1

    # [NOTE] Se um vertice terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    distance[seed] = 0 # definimos o nivel da raiz como 0
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # stack: pilha com os indices dos vertices que precisam ser investigados
    #   |_ stack[-1]: o vertice a ser investigado no momento
    #
    stack = [seed]

    while (len(stack) != 0):
        current = stack[-1] # pegamos o indice do vertice sendo investigado
        stack.pop(-1) # retiramos ele da pilha
        
        if not visited[current]:
            visited[current] = True

            # [REF] https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
            for neighbor in graph[current][::-1]: # pegamos todos os elementos da lista usando step=-1
                # seguindo o padrao da aula 5 (slide 18) percorremos os vertices em ordem descrescente
                # desse modo os vizinhos de menor indice ficam sempre no topo da pilha, isto eh, sao analisados primeiro
                if not visited[neighbor]: # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                    stack.append(neighbor) # adicionamos o vizinho na pilha

                    distance[neighbor] = distance[current] + 1 # o nivel do noh neighbor eh um nivel acima do atual
                    parent[neighbor] = current # o pai do noh neighbor eh o no sendo analisado

    return (distance, parent)

# -----------------------------------------------


def breadth_search(graph, seed=0, savefile=None):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')


    if graph.mem_mode == 'mtx':
        distance, parent = breadth_search_mtx(graph, seed)
    else:
        distance, parent = breadth_search_lst(graph, seed)


    if savefile:
        with open(savefile,'w') as f:
            for v in range(len(graph)):
                f.write('{},{}\n'.format(parent[v], distance[v]))

    return distance, parent

# -----------------------------------------------


def depth_search(graph, seed=0, savefile=None):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    if graph.mem_mode == 'mtx':
        distance, parent = depth_search_mtx(graph,seed)
    else:
        distance, parent = depth_search_lst(graph,seed)

    if savefile:
        with open(savefile,'w') as f:
            for v in range(len(graph)):
                f.write('{},{}\n'.format(parent[v], distance[v]))

    return distance, parent

# -----------------------------------------------

bfs = breadth_search
dfs = depth_search

# -----------------------------------------------


def DFSUtil(graph, temp, v, visited):
    visited[v] = True
 
    temp.append(v)
 
    for i in graph[v]:
        if visited[i] == False:
            temp = DFSUtil(graph, temp, i, visited)
    return temp

def connectedComponents(graph, vertices):
        visited = []
        cc = []
        for i in range(vertices):
            visited.append(False)
        for v in range(vertices):
            if visited[v] == False:
                temp = []
                cc.append(DFSUtil(graph, temp, v, visited))
        return cc

# -----------------------------------------------


def distance_mtx(graph, seed, target):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # [NOTE] Se um vertice terminar com distance[v] == -1, significa que ele nao esta conectado a raiz

    distance[seed] = 0 # definimos o nivel da raiz como 0

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = [seed]

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue.pop(0) # retiramos ele da fila

        for vert in range(len(graph)): # iremos percorrer todos os vertices para ver se sao vizinhos do atual
            if graph[current][vert]: # existe uma aresta (current,vert), isto eh, eles sao vizinhos
                if vert != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                    if not visited[vert]: # vert ainda nao foi investigado
                        queue.append(vert) # adicionamos vert a fila para ser investigado
                        visited[vert] = True # indicamos que vert ja foi descoberto
                        
                        distance[vert] = distance[current] + 1 # declaramos vert como estando 1 nivel acima de current

                        if vert == target:
                            return distance[target]

    return distance[target]

# -----------------------------------------------


def distance_lst(graph, seed, target):
    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    #
    visited = np.zeros(len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    #
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1

    # [NOTE] Se um vertice terminar com distance[v] == -1, significa que ele nao esta conectado a raiz

    distance[seed] = 0 # definimos o nivel da raiz como 0

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = [seed]

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue.pop(0) # retiramos ele da fila

        for neighbor in graph[current]: # verificamos todos os vizinhos do vertice sendo investigado
            if neighbor != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[neighbor]: # o vizinho ainda nao foi investigado
                    queue.append(neighbor) # adicionamos o vizinho a lista para ser investigado
                    visited[neighbor] = True # indicamos que o vizinho ja foi descoberto
        
                    distance[neighbor] = distance[current] + 1 # declaramos neighbor como estando 1 nivel acima de cur

                    if neighbor == target:
                        return distance[target]

    return distance[target]

# -----------------------------------------------


def distance(graph, seed, target):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    if graph.mem_mode =='mtx':
        return distance_mtx(graph, seed, target)
    else:
        return distance_lst(graph, seed, target)

# -----------------------------------------------


def diameter(graph):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    diameter = 0

    for v in range(len(graph)):
            distances = breadth_search(graph, seed=v)[0] # pegando os tamanhos de todos os caminhos simples que saem de v
            longest = distances.max() # pegando o tamanho do maior caminho simples que sai de v

            if longest > diameter:
                diameter = longest

    return diameter   

# -----------------------------------------------

def dijkstra(graph, s):

    dist = [ np.inf for v in range(len(graph)) ]
    dist[s] = 0

    parent = [ -1 for v in range(len(graph)) ]
    parent[s] = s

    # O heap ponderado, onde cada elemento tem uma chave identificadora e o peso que define 
    # sua ordem de prioridade. Usaremos os indices dos vertices como as chaves e as distancias 
    # ate a semenete como seus pesos
    heap = weighted_heap()

    for v in range(len(graph)):
        heap.insert(v, np.inf) if (v != s) else heap.insert(s, 0)

    u = heap.remove()
    while u is not None:
        u = u[0]
        for v in graph.neighbors(u):
            e = graph.edge(u,v)

            if dist[v] > dist[u] + e:
                parent[v] = u
                dist[v] = dist[u] + e
                heap.update(v, dist[v])

        u = heap.remove()

    return dist, parent

# -----------------------------------------------

