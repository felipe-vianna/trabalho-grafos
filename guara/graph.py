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
        self.adjac = np.array(0) # a matriz/lista de ajacencia
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

        self.shape['v'] = int(lines[0].split()[0]) # numero de vertices

        if self.weighted:
            self.adjac = np.zeros( (self.shape['v'], self.shape['v']) ) # inicializando a matriz com zeros (False)
        else:
            self.adjac = np.zeros( (self.shape['v'], self.shape['v']), dtype=np.uint8) # inicializando a matriz com zeros (False)
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

            if not self.adjac[ edge[0] ][ edge[1] ]: # verificamos se a aresta ja foi analisada e incluida no grafo

                self.adjac[ edge[0] ][ edge[1] ] = True if not self.weighted else edge[2]

                self.degrees[ edge[0] ] += 1
                self.shape['e'] += 1

                if not self.directed:
                    # quando o grafo eh nao-direcionado, a aresta (u,v) == (v,u)
                    self.adjac[ edge[1] ][ edge[0] ] = True if not self.weighted else edge[2]

                    self.degrees[ edge[1] ] += 1  

        return self.adjac

    # -----------------------------------------------

    def read_as_list(self, filename):

        # Lendo o arquivo
        lines = []
        with open(filename) as f:
            lines = f.readlines() # lista de strings contendo o conteudo de cada linha

        self.shape['v'] = int(lines[0]) # numero de vertices

        if not self.weighted:
            self.adjac = [ [] for i in range(self.shape['v']) ] # adjac[v] contem a lista de vizinhos do vertice v
        else:
            # adjac[v] contem uma lista onde cada elemento eh uma dupla do tipo (u,w), em que 
            # 'u' eh o indice do vertice vizinho de v e 'w' eh o peso da aresta incidente a eles
            self.adjac = [ [] for i in range(self.shape['v']) ]

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


            if not edge[1] in self.adjac[ edge[0] ]: # verificamos se a aresta ja foi analisada e incluida no grafo
                if not self.weighted:
                    self.adjac[ edge[0] ].append( edge[1] )
                else:
                    self.adjac[ edge[0] ].append( [edge[1],edge[2]] )

                self.degrees[ edge[0] ] += 1
                self.shape['e'] += 1

                if not self.directed:
                    # como o grafo eh nao direcionado, a aresta (i,j) eh equivalente a (j,i) e incluiremos ambas
                    # self.adjac[ edge[1] ].append( edge[0] if not self.weighted else [edge[0], edge[2]] )
                    if not self.weighted:
                        self.adjac[ edge[1] ].append( edge[0] )
                    else:
                        self.adjac[ edge[1] ].append( [edge[0],edge[2]] )
                    self.degrees[ edge[1] ] += 1

        self.adjac = np.array(self.adjac, dtype=object) # passando de lista para array numpy
        # agora ordenamos as listas de vizinhos em ordem crescente de indices (a lista de pesos eh ordenada em concordancia com essa ordem)
        for v in range(len(self.adjac)):
            if not self.weighted:
                self.adjac[v] = np.sort(self.adjac[v], axis=0) # ordenamos a lista de vizinhos de cada vertice
            else:
                self.adjac[v] = np.array(self.adjac[v])

                neighbors = self.adjac[v][:,0] # lista (array) dos vizinhos de v
                weights = self.adjac[v][:,1] # lista (array) dos pesos de cada aresta incidente a v

                # # [REF] https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
                neighbors_sort_index = np.argsort(neighbors) # os indices a serem usados para ordenar a lista de vizinhos de v

                self.adjac[v][:,0] = neighbors[neighbors_sort_index] # ordenamos a lista de vizinhos do menor para o maior
                self.adjac[v][:,1] = weights[neighbors_sort_index] # ordenamos os pesos de acordo com a lista de vizinhos


        return self.adjac

    # -----------------------------------------------

    def neighbors(self, vert):
        """
        Recebe um indice de vertice e retorna uma lista (array numpy) com os indices dos vertices 
        vizinhos a ele
        """
        if self.mem_mode == 'mtx':
            return np.where( self.adjac[vert] != 0 )[0] # selecionamos, na linha referente ao vertice, os indices dos elementos que sao nao-nulos
        else:
            if not self.weighted:
                return self.adjac[vert]
            else:
                # return self.adjac[vert][0] # pegamos, da linha do vertice, apenas a coluna referente ao indice dos vizinhos (e nao a dos pesos das arestas)
                return np.int32(self.adjac[vert][:,0]) # pegamos, da linha do vertice, apenas a coluna referente ao indice dos vizinhos (e nao a dos pesos das arestas)

    # -----------------------------------------------

    def weights(self, vert):
        """
        Recebe um indice de vertice e retorna uma lista (array numpy) com os pesos das arestas incidentes
        a ele. A ordem dos pesos eh equivalente a ordem crescente dos indices dos vizinhos. Isto, eh,
        o primeiro peso da lista eh referente ao vizinho de menor indice e assim por diante.
        """
        if self.mem_mode == 'mtx':
            return self.adjac[vert][ self.adjac[vert] != 0 ] # selecionamos, na linha referente ao vertice, os elementos nao-nulos
        else:
            if not self.weighted:
                return np.ones(self.adjac[vert])
            else:
                return self.adjac[vert][:,1] # pegamos, da linha do vertice, apenas a coluna referente aos pesos das arestas incidentes

    # -----------------------------------------------

    def edge(self, v1, v2):
        """
        Recebe dois indices de vertices e retorna o peso da aresta que liga eles (ou zero, caso os
        vertices nao sejam vizinhos)
        """
        if self.mem_mode == 'mtx':
            return self.adjac[v1][v2]
        else:
            if not self.weighted:
                # Podemos usar a funcao np.searchsorted() para percorrer a lista de vizinhos, que eh 
                # ordenada. Esta funcao tem complexidade O(log(n)), por isso eh uma alternativa
                # mais interessante que percorrer a lista inteira buscando v2
                where = np.searchsorted(self.adjac[v1], v2)
                if (where == len(self.adjac[v1])) or (v2 != self.adjac[v1][where]):
                    # v2 nao esta na lista dos vizinhos
                    # se where == len: v2 maior que todos os vizinhos
                    return 0

                return self.adjac[v1][where]
            else:
                where = np.searchsorted(self.adjac[v1][:,0], v2)
                if (where == len(self.adjac[v1])) or (v2 != self.adjac[v1][where,0]):
                    # v2 nao esta na lista dos vizinhos
                    # se where == len: v2 maior que todos os vizinhos
                    return 0

                return self.adjac[v1][where,1]

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
        s +=  '  ' + np.array2string(self.adjac, prefix='  ')
        return s

    # A funcao __getitem__() eh chamada quando usamos classe[key]
    def __getitem__(self, key):
        return self.adjac[key]

    def __iter__(self):
        return iter(range(self.n))


# -----------------------------------------------