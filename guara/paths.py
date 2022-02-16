
import numpy as np
from guara.graph import Graph
from guara.utils import weighted_heap

# -----------------------------------------------


def breadth_search(graph, seed=0, savefile=None):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    # [NOTE] Se um vertice v terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    visited = np.zeros(len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1
    distance[seed] = 0 # definimos o nivel da raiz como 0

    # parent[v]: o vertice pai de v na arvore gerada pela busca
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # queue: fila dos vertices que precisam ser investigados
    queue = [seed]

    while (len(queue) != 0):
        v = queue[0] # pegamos o vertice do inicio da fila
        queue.pop(0) # retiramos ele da fila

        for u in graph.neighbors(v):
            if u != v: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[u]: # o vizinho u ainda nao foi investigado
                    queue.append(u) # adicionamos o vizinho na fila para ser investigado
                    visited[u] = True # indicamos que o vizinho ja foi descoberto

                    distance[u] = distance[v] + 1 # declaramos o vizinho como estando 1 nivel acima do atual
                    parent[u] = v # declaramos o vertice atual como o pai de u na arvore


    if savefile:
        with open(savefile,'w') as f:
            for v in range(len(graph)):
                f.write('{},{}\n'.format(parent[v], distance[v]))

    return distance, parent

# -----------------------------------------------


def depth_search(graph, seed=0, savefile=None):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    # [NOTE] Se um vertice v terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    visited = np.zeros(graph.n, dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    distance = np.full(len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1
    distance[seed] = 0 # definimos o nivel da raiz como 0

    # parent[v]: o vertice pai de v na arvore gerada pela busca
    parent = np.full(len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # stack: pilha dos vertices que precisam ser investigados
    stack = [seed]

    while (len(stack) != 0):
        u = stack[-1] # pegamos o vertice do topo da pilha
        stack.pop(-1) # retiramos ele da pilha
        
        if not visited[u]:
            visited[u] = True

            # [REF] https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
            for v in graph.neighbors(u)[::-1]:
                # seguindo o padrao da aula 5 (slide 18) percorremos os vizinhos em ordem descrescente
                # assim os vizinhos de menor indice ficam no topo da pilha para serem investigados primeiro
                if not visited[v]: # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                    stack.append(v)

                    distance[v] = distance[u] + 1 # o nivel do noh neighbor eh um nivel acima do atual
                    parent[v] = u # o pai do noh neighbor eh o no sendo analisado

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

    dist = np.array( [ np.inf for v in range(len(graph)) ] )
    dist[s] = 0

    parent = np.array( [ -1 for v in range(len(graph)) ] )
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

# def floyd_warshall(graph):

#     # dist[i,j]: contem o custo total do menor caminho que se conhece que vai de i para j
#     dist = np.full( shape=(graph.n, graph.n), fill_value=np.inf)

#     neg_cycle = False # flag indicando se foi encontrado um ciclo negativo no grafo

#     # inicializamos dist com o custo para ir de um vertice a outro sem vertices intermediarios
#     for v in graph:
#         dist[v][ graph.neighbors(v) ] = graph.weights(v) # selecionamos, na linha v, os elementos correpospondentes aos vizinhos de v
#         dist[v,v] = 0 # o custo de ir de v para v eh nulo

#     # parent[i,j]: contem o vertice anterior a j no menor caminho que se conhece de i para j (isto eh, o pai de j)
#     parent = np.full( shape=(graph.n, graph.n), fill_value=-1)

#     for v in graph:
#         parent[v][ graph.neighbors(v) ] = v
#         parent[v,v] = v

#     for k in range(1, graph.n):
#         # comecamos o k em 1, pois k = 0 eh a propria inicializacao da matriz
#         # print(f'k:{k-1}')
#         # print(dist)
#         # print(parent)
#         for i in range(graph.n):
#             for j in range(graph.n):
#                 if (dist[i,k] + dist[k,j]) < dist[i,j]:

#                     dist[i,j] = (dist[i,k] + dist[k,j])
#                     parent[i,j] = parent[i,k]

#                     if (i == j) and (dist[i,i] < 0):
#                         neg_cycle = True
#                         # print(f'i:{i}, j:{j}, k:{k}')
#                         # print(dist)
#                         # print(parent)
#                         # raise Exception(f'Encontrado ciclo negativo no grafo {graph.name}. Nao foi possivel calcular distancias.')

#     return dist, parent, neg_cycle

# -----------------------------------------------


def mst(graph, s, savefile=None):
    """
    Recebe um  grafo nao-direcionado e um vertice semente para o algoritmo de Prim, retornando
    a MST resultante.

    A MST eh retornada como uma tupla (parent, cost), onde parent e cost sao listas no formato:
        parent[v]: o vert 'u' da aresta (u,v) presente na MST (para v != s)
        cost[v]: o peso da aresta (u,v) descrita por parent[v] (para v != s)
        parent[s] == s
        cost[s] == 0
    Para obter o custo total da arvore, pode-se usar o metodo cost.sum()

    Caso o grafo seja desconexo, nao existe (por definicao) uma MST. Nesse caso eh retornada a
    tupla (None, None).

    Caso seja especificado um arquivo em savefile, o arquivo eh no formato:
        primeira linha: '<numero_de_vertices> <custo_total_mst>'
        demais linhas: '<u> <v> <peso_uv>'
    """

    cost = np.array([ np.inf for v in graph ])
    cost[s] = 0

    parent = np.array([ -1 for v in graph ]) # nesse algoritmo, o pai de um vertice eh o no que o inseriu na arvore
    parent[s] = s

    # O heap ponderado, onde cada elemento tem uma chave identificadora e o peso que define 
    # sua ordem de prioridade. Usaremos os indices dos vertices como as chaves e as distancias 
    # ate a semenete como seus pesos
    heap = weighted_heap()

    for v in graph:
        heap.insert(v, np.inf) if (v != s) else heap.insert(s, 0)

    u = heap.remove() # tiramos a semente do heap

    while u is not None:
        u = u[0] #lembrando que cada elemento no heap contem chave e peso, pegamos a chave (vert)

        for v in graph.neighbors(u):
            e = graph.edge(u,v)

            if (cost[v] > e) and (parent[u] != v):
                # Ao percorrermos os vizinhos de u, ignoramos a aresta (v,u) que ja tenha
                # sido incluida na arvore. Se v for pai de u, eh porque ele ja estava na 
                # arvore e u entrou por meio dele.
                parent[v] = u
                cost[v] = e
                heap.update(v, cost[v])
                # print(u, v, parent, cost)

        u = heap.remove()

    if (-1 in parent): 
        # ha componentes desconexas no grafo, entao nao existe MST desse grafo
        # (por definicao a MST conecta todos os vertices do grafo)
        return None, None

    if savefile:
        with open(savefile,'w') as f:
            f.write(f'{len(cost)} {cost.sum()}')
            for v in range(len(graph)):
                if v != s:
                    f.write(f'{parent[v]} {v} {cost[v]}\n')

    return parent, cost