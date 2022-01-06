import os
import numpy as np

class Graph:
    def __init__(self, filename, memory_mode='mtx'):

        # Verificando se o nome do arquivo passado eh uma string
        if (type(filename) != type('')) or (not filename):
            print('Error: Graph expects a string with the path to the text file that describes the graph.')
            exit()

        # Verificando se o modo de armazenamento em memoria passado eh valido
        self.__mode = memory_mode.lower()
        if (self.__mode != 'mtx') and (self.__mode != 'lst'):
            print('Error: memory mode must be either \"mtx\" (matrix) or \"lst\" (list)')
            exit()

        # Verificando se o arquivo passado existe
        if not os.path.isfile(filename):
            print("Error: file not found.")
            print("--\t"+filename)

        # -----------------------------------------------

        self.name = os.path.split(filename)[1] # separando o nome do arquivo do diretorio
        self.name = os.path.splitext(self.name)[0] # separando o nome do arquivo e a extensao

        # -----------------------------------------------

        self.__shape = {'v': 0,'e': 0} # dicionario com o numero de vertices e arestas
        self.__graph = np.array(0)
        self.__degrees = np.array(0)

        self.__dg_min = None
        self.__dg_max = None
        self.__dg_avg = None
        self.__dg_median = None
        
        # -----------------------------------------------

        if self.__mode == 'mtx':
            self.read_as_matrix(filename)
        else:
            self.read_as_list(filename)

    # } __init__

    # -----------------------------------------------

    def read_as_matrix(self,filename):

        # Lendo o arquivo
        lines = []
        with open(filename) as f:
            lines = f.readlines() # lista de strings contendo o conteudo de cada linha

        self.__shape['v'] = int(lines[0]) # numero de vertices

        self.__graph = np.zeros((self.__shape['v'],self.__shape['v']), dtype=np.uint8) # inicializando a matriz com zeros (False)
        self.__degrees = np.zeros(self.__shape['v'], dtype=np.uint) # inicializando todos os graus dos vertices em zero

        self.__dg_min = None
        self.__dg_max = None
        self.__dg_avg = None
        self.__dg_median = None

        self.__shape['e'] = 0
        for edge in lines[1:]: # cada linha representa uma aresta

            edge = edge.split()

            edge[0] = int(edge[0]) - 1 # [NOTE] consideramos que os indices dos vertices no arquivo texto iniciam sempre em 1
            edge[1] = int(edge[1]) - 1


            if not self.__graph[ edge[0] ][ edge[1] ]: # verificamos se a aresta ja foi analisada e incluida no grafo
                # como o grafo eh nao direcionado, a aresta (i,j) eh equivalente a (j,i) e incluiremos ambas
                self.__graph[ edge[0] ][ edge[1] ] = True
                self.__graph[ edge[1] ][ edge[0] ] = True # podemos colocar essa linha pois o grafo eh nao-direcionado
                self.__shape['e'] += 1

                self.__degrees[ edge[0] ] += 1
                self.__degrees[ edge[1] ] += 1

        # v = 0
        # self.__dg_max = self.__degrees[0]
        # self.__dg_min = self.__degrees[0]
        # for d in self.__degrees:
        #     v += 1

        #     self.__dg_avg += d

        #     self.__dg_max = d if d > self.__dg_max
        #     self.__dg_min = d if d < self.__dg_min    

        return self.__graph

    # -----------------------------------------------

    def read_as_list(self, filename):

        # Lendo o arquivo
        lines = []
        with open(filename) as f:
            lines = f.readlines() # lista de strings contendo o conteudo de cada linha

        self.__shape['v'] = int(lines[0]) # numero de vertices

        self.__graph = np.empty(self.__shape['v'], dtype=object) # inicializando a lista/array de listas de vizinhos
        for v in range(self.__shape['v']):
            self.__graph[v] = [] # inicializando as listas de vizinhos com um array vazio

        self.__degrees = np.zeros(self.__shape['v'], dtype=np.uint) # inicializando todos os graus dos vertices em zero

        self.__dg_min = None
        self.__dg_max = None
        self.__dg_avg = None
        self.__dg_median = None

        self.__shape['e'] = 0

        for edge in lines[1:]: # cada linha representa uma aresta

            edge = edge.split()

            edge[0] = int(edge[0]) - 1 # [NOTE] consideramos que os indices dos vertices no arquivo texto iniciam sempre em 1
            edge[1] = int(edge[1]) - 1


            if not edge[1] in self.__graph[ edge[0] ]: # verificamos se a aresta ja foi analisada e incluida no grafo
                # como o grafo eh nao direcionado, a aresta (i,j) eh equivalente a (j,i) e incluiremos ambas
                self.__graph[ edge[0] ].append(edge[1])

                self.__graph[ edge[1] ].append(edge[0])
                self.__shape['e'] += 1

                self.__degrees[ edge[0] ] += 1
                self.__degrees[ edge[1] ] += 1

        for v in range(len(self.graph)):
            self.graph[v] = np.sort(self.graph[v])

        return self.__graph

    # -----------------------------------------------

    """
        [NOTE] Os atributos que possuem dois underline (__) a frente do nome sao como se fossem privados
        (ainda podem ser acessados, mas ficam mais escondidos). Usamos o decorador 'property' para torna-los
        facilmente visiveis para o usuario.
        
        - https://www.tutorialsteacher.com/python/public-private-protected-modifiers
        - https://www.tutorialsteacher.com/python/property-decorator
        - https://www.programiz.com/python-programming/property
    """

    @property
    def mode(self):
        return self.__mode

    @property
    def graph(self):
        return self.__graph

    @property
    def shape(self):
        return self.__shape
        
    @property
    def n(self):
        return self.__shape['v']

    @property
    def m(self):
        return self.__shape['e']

    @property
    def degrees(self):
        return self.__degrees
    
    @property
    def dg_min(self):
        if (self.__dg_min is None):
            self.__dg_min = self.__degrees.min()
        return self.__dg_min
    
    @property
    def dg_max(self):
        if (self.__dg_max is None):
            self.__dg_max = self.__degrees.max()
        return self.__dg_max

    @property
    def dg_avg(self):
        if (self.__dg_avg is None):
            self.__dg_avg = 2 * self.__degrees.mean() # de acordo com definicao do slide (aula 3 - slide 10)
        return self.__dg_avg

    @property
    def dg_median(self):
        if (self.__dg_median is None):
            self.__dg_median = np.median(self.__degrees)
        return self.__dg_median

    # A funcao __len__() eh chamada quando usamos len(classe)
    def __len__(self):
        return self.__shape['v']

    # A funcao __repr__() eh chamada quando usamos print(classe)
    def __repr__(self):
        s = f'Graph \"{self.name}\" ({self.__mode})\n'
        s += '  {}\n'.format(self.__shape)
        s += '  ' + np.array2string(self.__graph, prefix='  ')
        return s

    def __getitem__(self, key):
        return self.__graph[key]


# -----------------------------------------------


def breadth_search_as_mtx(graph, seed=0):
    # visited: lista de tamanho n com flags indicando se um vertice ainda precisa ser investigado
    #
    visited = np.zeros(graph.n, dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # tree: array representando a arvore gerada. cada elemento contem uma dupla do tipo [pai, nivel]
    #   representando o pai e o nivel na arvore do noh v do grafo
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype=np.int32 ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = np.array([seed], dtype=np.int32)

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue = queue[1:] # retiramos ele da fila

        for vert in range(graph.n): # iremos percorrer todos os vertices para ver se sao vizinhos do atual
            if graph.graph[current][vert]: # existe uma aresta (current,vert), isto eh, eles sao vizinhos
                if vert != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                    if not visited[vert]: # vert ainda nao foi investigado
                        queue = np.append(queue, np.int32(vert)) # adicionamos vert a fila para ser investigado
                        visited[vert] = True # indicamos que vert ja foi descoberto
                        
                        tree[vert][0] = current # declaramos current como o vertice pai de vert na arvore
                        tree[vert][1] = tree[current][1] + 1 # declaramos vert como estando 1 nivel acima de current

    return tree


def breadth_search_as_lst(graph, seed=0):
    # visited: lista de tamanho n com flags indicando se um vertice ainda precisa ser investigado
    #
    visited = np.zeros(graph.n, dtype=np.int8)
    visited[seed] = False # iniciamos a descoberta dos vertices pela semente

    # tree: array representando a arvore gerada. cada elemento contem uma dupla do tipo [pai, nivel]
    #   representando o pai e o nivel na arvore do noh v do grafo
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype=np.int32 ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = np.array([seed], dtype=np.int32)

    while (len(queue) != 0):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue = queue[1:] # retiramos ele da fila

        for neighbor in graph.graph[current]: # verificamos todos os vizinhos do vertice sendo investigado
            if neighbor != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[neighbor]: # o vizinho ainda nao foi investigado
                    queue = np.append(queue, np.int32(neighbor)) # adicionamos o vizinho a lista para ser investigado
                    visited[neighbor] = True # indicamos que o vizinho ja foi descoberto
        
                    tree[neighbor][0] = current # declaramos cur como o vertice pai de neighbor na arvore
                    tree[neighbor][1] = tree[current][1] + 1 # declaramos neighbor como estando 1 nivel acima de cur

    return tree


def breadth_search(graph, seed=0, filename=None):
    # if type(graph) != Graph:
    #     print('Error: graph must be of class Graph')
    #     exit()


    if graph.mode == 'mtx':
        tree = breadth_search_as_mtx(graph, seed)
    else:
        tree = breadth_search_as_lst(graph, seed)


    if filename:
        with open(filename,'w') as f:
            for v in tree:
                f.write('{},{}\n'.format(v[0], v[1]))

    return tree

# -----------------------------------------------


def depth_search_as_mtx(graph, seed=0):
    visited = np.zeros(graph.n, dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    stack = np.array([seed], dtype=np.int32)

    # tree: array representando a arvore gerada. cada elemento tree[v] eh uma dupla do tipo [pai, nivel]
    #   representando o pai e o nivel na arvore do noh v do grafo
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype= np.int32 ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    while (len(stack) != 0):
        current = stack[-1] # analisamos o elemento no topo da pilha

        stack = np.delete(stack, -1) # retiramos o elemento do topo da pilha
        
        if not visited[current]:
            visited[current] = True

            # [REF] https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
            for vert in graph.graph[current][::-1]: # pegamos todos os elementos da lista usando step=-1
                # seguindo o padrao da aula 5 (slide 18) percorremos os vertices em ordem descrescente
                # desse modo os vizinhos de menor indice ficam sempre no topo da pilha, isto eh, sao analisados primeiro
                if (graph.graph[current][vert]): # se os dois forem vizinhos
                    if (not visited[vert]): # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                        stack = np.append(stack, np.int32(vert)) # adicionamos o vizinho na pilha

                        tree[vert][0] = current # o pai do noh vert eh o no sendo analisado
                        tree[vert][1] = tree[current][1] + 1 # o nivel do noh  vert eh um nivel acima do atual

    return tree


def depth_search_as_lst(graph, seed=0):
    visited = np.zeros(graph.n, dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    stack = np.array([seed], dtype=np.int32)

    # tree: array representando a arvore gerada. cada elemento tree[v] eh uma dupla do tipo [pai, nivel]
    #   representando o pai e o nivel na arvore do noh v do grafo
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype= np.int32 ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    while (len(stack) != 0):
        current = stack[-1] # analisamos o elemento no topo da pilha

        stack = stack[:-1] # retiramos o elemento do topo da pilha
        
        if not visited[current]:
            visited[current] = True

            # [REF] https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
            for neighbor in graph.graph[current][::-1]: # pegamos todos os elementos da lista usando step=-1
                # seguindo o padrao da aula 5 (slide 18) percorremos os vertices em ordem descrescente
                # desse modo os vizinhos de menor indice ficam sempre no topo da pilha, isto eh, sao analisados primeiro
                if (not visited[neighbor]): # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                    stack = np.append(stack, np.int32(neighbor)) # adicionamos o vizinho na pilha

                    tree[neighbor][0] = current # o pai do noh vert eh o no sendo analisado
                    tree[neighbor][1] = tree[current][1] + 1 # o nivel do noh  vert eh um nivel acima do atual

    return tree


def depth_search(graph, seed=0, filename=None):
    # if type(graph) != Graph:
    #     print('Error: graph must be of class Graph')
    #     exit()

    if graph.mode == 'mtx':
        tree = depth_search_as_mtx(graph,seed)
    else:
        tree = depth_search_as_lst(graph,seed)

    if filename:
        with open(filename,'w') as f:
            for v in tree:
                f.write('{},{}\n'.format(v[0], v[1]))

    return tree

bfs = breadth_search
dfs = depth_search

def distance_as_lst_2(graph, seed, destiny):
    # visited: lista de tamanho n com flags indicando se um vertice ainda precisa ser investigado
    #
    visited = np.zeros(graph.n, dtype=np.int8)
    visited[seed] = False # iniciamos a descoberta dos vertices pela semente

    # distance[v]: distancia (medida em arestas) do vertice v ate a semente. se um vertice
    #    terminar com distancia == -1, significa que ele nao esta conectado a semente (grafo desconexo)
    #
    distance = np.ones( graph.n, dtype=np.int32 ) * (-1)

    distance[seed] = 0 # definimos a semente como tendo distancia 0 de si mesma
    
    # queue: fila com os indices dos vertices que precisam ser investigados
    #   |_ queue[0]: o vertice a ser investigado no momento
    #
    queue = np.array([seed])
    current = seed

    while (len(queue) != 0) and (current != destiny):
        current = queue[0] # pegamos o indice do vertice sendo investigado
        queue = queue[1:] # retiramos ele da fila

        for neighbor in graph.graph[current]: # verificamos todos os vizinhos do vertice sendo investigado
            if neighbor != current: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[neighbor]: # o vizinho ainda nao foi investigado
                    queue = np.append(queue, np.int32(neighbor)) # adicionamos o vizinho a lista para ser investigado
                    visited[neighbor] = True # indicamos que o vizinho ja foi descoberto
        
                    distance[neighbor] = distance[current] + 1 # declaramos neighbor como estando 1 aresta a mais do vertice atual

    return distance[destiny]

def distance_2(graph, seed, destiny):
    if graph.mode == 'mtx':
        print('aviso: distance_as_mtx nao implementada ainda')
        d = -1
    else:
        d = distance_as_lst_2(graph, seed, destiny)

    return d

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

def distance(graph, seed, target, v, pred, dist):
	queue = []
	visited = [False for i in range(v)]
	for i in range(v):
		dist[i] = 1000000
		pred[i] = -1

	visited[seed] = True
	dist[seed] = 0
	queue.append(seed)

	while (len(queue) != 0):
		u = queue[0]
		queue.pop(0)
		for i in range(len(graph[u])):
		
			if (visited[graph[u][i]] == False):
				visited[graph[u][i]] = True
				dist[graph[u][i]] = dist[u] + 1
				pred[graph[u][i]] = u
				queue.append(graph[u][i])

				if (graph[u][i] == target):
					return True

	return False

def shortestDistance(adj, s, dest, v):
	pred=[0 for i in range(v)]
	dist=[0 for i in range(v)]
	if (distance(adj, s, dest, v, pred, dist) == False):
		return -1
	path = []
	crawl = dest
	crawl = dest
	path.append(crawl)
	
	while (pred[crawl] != -1):
		path.append(pred[crawl])
		crawl = pred[crawl]
	return dist[dest]
		
def diameter(graph):
    diameter = 0
    for v in range(len(graph)):
        for v2 in range(len(graph)):
            shortest = shortestDistance(graph, v, v2, len(graph))
            if shortest > diameter:
                diameter = shortest
    return diameter   

def writeOnFile(filename, info):
    if filename:
        with open(filename,'w') as f:
            for v in info:
                f.write('{}: {}\n'.format(v[0], v[1]))

# MAIN
# -----------------------------------------------
