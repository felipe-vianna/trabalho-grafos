import os
import numpy as np

class Graph:
    def __init__(self, filename, memory_mode='mtx'):

        # Verificando se o nome do arquivo passado eh uma string
        if (type(filename) != type('')):
            raise TypeError('Graph expects a string with the path to the text file that describes the graph')


        # Verificando se o modo de armazenamento em memoria passado eh valido
        self.__mode = memory_mode.lower()
        if (self.__mode != 'mtx') and (self.__mode != 'lst'):
            raise ValueError('memory_mode must be either \"mtx\" (matrix) or \"lst\" (list)')

        # Verificando se o arquivo passado existe
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"file not found \"{filename}\"")

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

    # A funcao __getitem__() eh chamada quando usamos classe[key]
    def __getitem__(self, key):
        return self.__graph[key]


# -----------------------------------------------


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


    if graph.mode == 'mtx':
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

    if graph.mode == 'mtx':
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

    if graph.mode =='mtx':
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


def writeOnFile(filename, info):
    if filename:
        with open(filename,'w') as f:
            for v in info:
                f.write('{}: {}\n'.format(v[0], v[1]))


# MAIN
# -----------------------------------------------
