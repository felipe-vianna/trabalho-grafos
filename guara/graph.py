import os
import numpy as np

class Graph:
    def __init__(self, filename, mem_mode='lst', weighted=False, directed=False):

        # Verificando se o nome do arquivo passado eh uma string
        if (type(filename) != type('')):
            raise TypeError('Graph expects a string with the path to the text file that describes the graph')


        # Verificando se o modo de armazenamento em memoria passado eh valido
        self.mem_mode = mem_mode.lower()
        if (self.mem_mode != 'mtx') and (self.mem_mode != 'lst'):
            raise ValueError('memory_mode must be either \"mtx\" (matrix) or \"lst\" (list)')

        # Verificando se o arquivo passado existe
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"file not found \"{filename}\"")

        # -----------------------------------------------

        self.name = os.path.split(filename)[1] # separando o nome do arquivo do diretorio
        self.name = os.path.splitext(self.name)[0] # separando o nome do arquivo e a extensao

        # -----------------------------------------------

        self.shape = {'v': 0,'e': 0} # dicionario com o numero de vertices (v) e arestas (e)
        self.graph = np.array(0) # a matriz/lista de ajacencia
        self.degrees = np.array(0) # degree[v]: o grau do vertice v
        
        # -----------------------------------------------

        self.weighted = weighted
        self.has_neg_weights = False
        self.directed = directed
        
        # -----------------------------------------------

        if self.mem_mode == 'mtx':
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

        self.shape['v'] = int(lines[0]) # numero de vertices

        if self.weighted:
            self.graph = np.zeros( (self.shape['v'], self.shape['v']) ) # inicializando a matriz com zeros (False)
        else:
            self.graph = np.zeros( (self.shape['v'], self.shape['v']), dtype=np.uint8) # inicializando a matriz com zeros (False)
        self.degrees = np.zeros(self.shape['v']) # inicializando todos os graus dos vertices em zero

        self.shape['e'] = 0
        for edge in lines[1:]: # cada linha representa uma aresta

            edge = edge.split() # separando a linha em duas strings contendo, cada uma, um dos vertices da aresta

            edge[0] = int(edge[0]) - 1 # [NOTE] consideramos que os indices dos vertices no arquivo texto iniciam sempre em 1
            edge[1] = int(edge[1]) - 1
            if self.weighted:
                edge[2] = float(edge[2]) # o peso da aresta
                if edge[2] < 0:
                    self.has_neg_weights = True

            if not self.graph[ edge[0] ][ edge[1] ]: # verificamos se a aresta ja foi analisada e incluida no grafo

                self.graph[ edge[0] ][ edge[1] ] = True if not self.weighted else edge[2]

                self.degrees[ edge[0] ] += 1
                self.shape['e'] += 1

                if not self.directed:
                    # quando o grafo eh nao-direcionado, a aresta (u,v) == (v,u)
                    self.graph[ edge[1] ][ edge[0] ] = True if not self.weighted else edge[2]

                    self.degrees[ edge[1] ] += 1  

        return self.graph

    # -----------------------------------------------

    def read_as_list(self, filename):

        # Lendo o arquivo
        lines = []
        with open(filename) as f:
            lines = f.readlines() # lista de strings contendo o conteudo de cada linha

        self.shape['v'] = int(lines[0]) # numero de vertices

        if not self.weighted:
            self.graph = [ [] for i in range(self.shape['v']) ] # graph[v] contem a lista de vizinhos do vertice v
        else:
            # graph[v] contem duas listas: uma de vizinhos de v e uma com os pesos das arestas que os ligam
            #   graph[v][0]: lista de vizinhos de v
            #   graph[v][1]: lista de pesos das arestas que liga v a seus vizinhos
            self.graph = [ [ [],[] ] for i in range(self.shape['v']) ]

        self.degrees = np.zeros(self.shape['v']) # inicializando todos os graus dos vertices em zero

        self.shape['e'] = 0

        for edge in lines[1:]: # cada linha representa uma aresta

            edge = edge.split() # separando a linha em duas strings contendo, cada uma, um dos vertices da aresta

            edge[0] = int(edge[0]) - 1 # [NOTE] consideramos que os indices dos vertices no arquivo texto iniciam sempre em 1
            edge[1] = int(edge[1]) - 1
            if self.weighted:
                edge[2] = float(edge[2])
                if edge[2] < 0:
                    self.has_neg_weights = True


            if not edge[1] in self.graph[ edge[0] ]: # verificamos se a aresta ja foi analisada e incluida no grafo
                # self.graph[ edge[0] ].append( edge[1] if not self.weighted else np.array([edge[1], edge[2]]) )
                if not self.weighted:
                    self.graph[ edge[0] ].append( edge[1] )
                else:
                    self.graph[ edge[0] ][0].append( edge[1] ) # adicionamos o indice do vertice edge[1], vizinho de edge[0]
                    self.graph[ edge[0] ][1].append( edge[2] ) # adicionamos o peso da aresta que liga edge[0] a edge[1]

                self.degrees[ edge[0] ] += 1
                self.shape['e'] += 1

                if not self.directed:
                    # como o grafo eh nao direcionado, a aresta (i,j) eh equivalente a (j,i) e incluiremos ambas
                    # self.graph[ edge[1] ].append( edge[0] if not self.weighted else [edge[0], edge[2]] )
                    if not self.weighted:
                        self.graph[ edge[1] ].append( edge[0] )
                    else:
                        self.graph[ edge[1] ][0].append( edge[0] ) # adicionamos o indice do vertice edge[0], vizinho de edge[1]
                        self.graph[ edge[1] ][1].append( edge[2] ) # adicionamos o peso da aresta que liga edge[0] a edge[1]
                    self.degrees[ edge[1] ] += 1

        self.graph = np.array(self.graph, dtype=object) # passando de lista para array numpy
        # agora ordenamos as listas de vizinhos em ordem crescente de indices (a lista de pesos eh ordenada em concordancia com essa ordem)
        for v in range(len(self.graph)):
            if not self.weighted:
                self.graph[v] = np.sort(self.graph[v], axis=0) # ordenamos a lista de vizinhos de cada vertice
            else:
                neighbors = np.array(self.graph[v][0])
                weights = np.array(self.graph[v][1])

                neighbors_sort_index = np.argsort(neighbors) # os indices a serem usados para ordenar a lista de vizinhos de v

                # [REF] https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
                self.graph[v][0] = np.take_along_axis(neighbors, neighbors_sort_index, axis=0) # ordenamos a lista de vizinhos de cada vertice
                self.graph[v][1] = np.take_along_axis(weights, neighbors_sort_index, axis=0) # ordenamos a lista de pesos de acordo com a ordem dos vizinhos

        return self.graph

    # -----------------------------------------------

    def neighbors(self, vert):
        """
        Recebe um indice de vertice e retorna uma lista (array numpy) com os indices dos vertices 
        vizinhos a ele
        """
        if self.mem_mode == 'mtx':
            return np.nonzero( self.graph[vert] != 0 ) # selecionamos, na linha referente ao vertice, os indices dos elementos que sao nao-nulos
        else:
            if not self.weighted:
                return self.graph[vert]
            else:
                return self.graph[vert][0] # pegamos, da linha do vertice, apenas a coluna referente ao indice dos vizinhos (e nao a dos pesos das arestas)

    # -----------------------------------------------

    def weights(self, vert):
        """
        Recebe um indice de vertice e retorna uma lista (array numpy) com os pesos das arestas incidentes
        a ele. A ordem dos pesos eh equivalente a ordem crescente dos indices dos vizinhos. Isto, eh,
        o primeiro peso da lista eh referente ao vizinho de menor indice e assim por diante.
        """
        if self.mem_mode == 'mtx':
            return self.graph[vert][ self.graph[vert] != 0 ] # selecionamos, na linha referente ao vertice, os elementos nao-nulos
        else:
            if not self.weighted:
                return np.ones(self.graph[vert][0])
            else:
                return self.graph[vert][1] # pegamos, da linha do vertice, apenas a coluna referente aos pesos das arestas incidentes

    # -----------------------------------------------

    def edge(self, v1, v2):
        """
        Recebe dois indices de vertices e retorna o peso da aresta que liga eles ou zero, caso os
        vertices nao sejam vizinhos
        """
        if self.mem_mode == 'mtx':
            return self.graph[v1][v2]
        else:
            if not self.weighted:
                return 1 if v2 in self.graph[v1] else 0
            else:
                for i, neighbor in enumerate(self.graph[v1][0]):
                    if neighbor == v2:
                        return self.graph[v1][1][i] # retornamos o peso
                    elif neighbor > v2: # a lista de vizinhos eh ordenada, portanto se o vizinho atual for maior que v2, quer dizer que v2 nao esta na lista
                        return 0

    # -----------------------------------------------

    """
        [REF]
        - https://www.tutorialsteacher.com/python/property-decorator
        - https://www.programiz.com/python-programming/property
    """
        
    @property
    def n(self):
        return self.shape['v']

    @property
    def m(self):
        return self.shape['e']

    # -----------------------------------------------

    def dg_min(self):
        return self.degrees.min()
    
    def dg_max(self):
        return self.degrees.max()

    def dg_avg(self):
        # de acordo com definicao do slide (aula 3 - slide 10)
        return 2 * (self.m / self.n)

    def dg_median(self):
        return np.median(self.degrees)

    # A funcao __len__() eh chamada quando usamos len(classe)
    def __len__(self):
        return self.shape['v']

    # A funcao __repr__() eh chamada quando usamos print(classe)
    def __repr__(self):
        s =  f'Graph \"{self.name}\" '
        s += f'(mem_mode: {self.mem_mode}, weighted: {self.weighted}, directed: {self.directed})\n'
        s += f'  {self.shape}\n'
        s +=  '  ' + np.array2string(self.graph, prefix='  ')
        return s

    # A funcao __getitem__() eh chamada quando usamos classe[key]
    def __getitem__(self, key):
        return self.graph[key]


# -----------------------------------------------



# def writeOnFile(filename, info):
#     if filename:
#         with open(filename,'w') as f:
#             for v in info:
#                 f.write('{}: {}\n'.format(v[0], v[1]))


# MAIN
# -----------------------------------------------
