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

        self.__dg_min = 0
        self.__dg_max = 0
        self.__dg_avg = 0.0
        self.__dg_median = 0.0
        
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

        self.__dg_min = 0
        self.__dg_max = 0
        self.__dg_avg = 0.0
        self.__dg_median = 0.0

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
        for v in np.arange(self.__shape['v']):
            self.__graph[v] = np.empty(0, dtype=np.uint) # inicializando as listas de vizinhos com um array vazio

        self.__degrees = np.zeros(self.__shape['v'], dtype=np.uint) # inicializando todos os graus dos vertices em zero

        self.__dg_min = 0
        self.__dg_max = 0
        self.__dg_avg = 0.0
        self.__dg_median = 0.0

        self.__shape['e'] = 0

        for edge in lines[1:]: # cada linha representa uma aresta

            edge = edge.split()

            edge[0] = int(edge[0]) - 1 # [NOTE] consideramos que os indices dos vertices no arquivo texto iniciam sempre em 1
            edge[1] = int(edge[1]) - 1


            if not edge[1] in self.__graph[ edge[0] ]: # verificamos se a aresta ja foi analisada e incluida no grafo
                # como o grafo eh nao direcionado, a aresta (i,j) eh equivalente a (j,i) e incluiremos ambas
                self.__graph[ edge[0] ] = np.append(self.__graph[ edge[0] ], edge[1])
                self.__graph[ edge[1] ] = np.append(self.__graph[ edge[1] ], edge[0]) # podemos colocar essa linha pois o grafo eh nao-direcionado
                self.__shape['e'] += 1

                self.__degrees[ edge[0] ] += 1
                self.__degrees[ edge[1] ] += 1

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
        # [TODO] modo ineficiente pois se __dg_min == 0, o calculo sera realizado toda vez que tentarmos acessar esse atributo
        if not self.__dg_min:
            self.__dg_min = self.__degrees.min()
        return self.__dg_min
    
    @property
    def dg_max(self):
        # [TODO] modo ineficiente pois se __dg_max == 0, o calculo sera realizado toda vez que tentarmos acessar esse atributo
        if not self.__dg_max:
            self.__dg_max = self.__degrees.max()
        return self.__dg_max

    @property
    def dg_avg(self):
        # [TODO] modo ineficiente pois se __dg_avg == 0, o calculo sera realizado toda vez que tentarmos acessar esse atributo
        if not self.__dg_avg:
            self.__dg_avg = 2 * self.__degrees.mean() # de acordo com definicao do slide (aula 3 - slide 10)
        return self.__dg_avg

    @property
    def dg_median(self):
        # [TODO] modo ineficiente pois se __dg_median == 0, o calculo sera realizado toda vez que tentarmos acessar esse atributo
        if not self.__dg_median:
            self.__dg_median = np.median(self.__degrees)
        return self.__dg_median

    # A funcao __len__() eh chamada quando usamos len(classe)
    def __len__(self):
        return self.__shape['v']

    # A funcao __repr__() eh chamada quando usamos print(classe)
    def __repr__(self):
        s = 'Graph \"' + self.name + '\"\n'
        s += '  {}\n'.format(self.shape)
        s += '  ' + np.array2string(self.__graph, prefix='  ')
        return s



def breadth_search_as_mtx(graph, seed):
    # undiscovered: lista de tamanho n com flags indicando se um vertice ainda precisa ser investigado
    #
    undiscovered = np.ones(graph.n, dtype=np.int8)
    undiscovered[seed] = False # iniciamos a descoberta dos vertices pela semente

    # tree: array representando a arvore gerada. cada elemento contem uma dupla do tipo [pai, nivel]
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype=int ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    # discovered: fila com os indices dos vertices que precisam ser investigados
    #   |_ discovered[0]: o vertice a ser investigado no momento
    #
    discovered = np.array([seed])

    while (len(discovered) != 0):
        cur = discovered[0] # pegamos o indice do vertice sendo investigado
        discovered = np.delete(discovered, 0) # retiramos ele da fila

        for v in np.arange(graph.n):
            if graph.graph[cur][v]: # existe uma aresta (cur,v), isto eh, eles sao vizinhos
                if v != cur: # para evitar loops, ignoramos arestas do tipo (i,i)
                    if undiscovered[v]: # v ainda nao foi investigado
                        discovered = np.append(discovered, v) # adicionamos v a lista para ser investigado
                        undiscovered[v] = False # indicamos que v ja foi descoberto
                        
                        tree[v][0] = cur # declaramos cur como o vertice pai de v na arvore
                        tree[v][1] = tree[cur][1] + 1 # declaramos v como estando 1 nivel acima de cur

    return tree


def breadth_search_as_lst(graph, seed):
    # undiscovered: lista de tamanho n com flags indicando se um vertice ainda precisa ser investigado
    #
    undiscovered = np.ones(graph.n, dtype=np.int8)
    undiscovered[seed] = False # iniciamos a descoberta dos vertices pela semente

    # tree: array representando a arvore gerada. cada elemento contem uma dupla do tipo [pai, nivel]
    #   caso algum vertice termine com [-1,-1], significa que ele nao esta conectado a arvore
    #
    tree = np.ones( (graph.n, 2), dtype=int ) * (-1)

    tree[seed][0] = seed # colocamos a raiz como pai de si mesma
    tree[seed][1] = 0 # definimos o nivel da raiz como 0

    # discovered: fila com os indices dos vertices que precisam ser investigados
    #   |_ discovered[0]: o vertice a ser investigado no momento
    #
    discovered = np.array([seed])

    while (len(discovered) != 0):
        cur = discovered[0] # pegamos o indice do vertice sendo investigado
        discovered = np.delete(discovered, 0) # retiramos ele da fila

        for neighbor in graph.graph[cur]: # verificamos todos os vizinhos do vertice sendo investigado
            if neighbor != cur: # para evitar loops, ignoramos arestas do tipo (i,i)
                if undiscovered[neighbor]: # o vizinho ainda nao foi investigado
                    discovered = np.append(discovered, neighbor) # adicionamos o vizinho a lista para ser investigado
                    undiscovered[neighbor] = False # indicamos que o vizinho ja foi descoberto
        
                    tree[neighbor][0] = cur # declaramos cur como o vertice pai de neighbor na arvore
                    tree[neighbor][1] = tree[cur][1] + 1 # declaramos neighbor como estando 1 nivel acima de cur

    return tree


def breadth_search(graph, seed=0, filename=None):
    if type(graph) != Graph:
        print('Error: graph must be of class Graph')
        exit()


    if graph.mode == 'mtx':
        tree = breadth_search_as_mtx(graph, seed)
    else:
        tree = breadth_search_as_lst(graph, seed)


    if filename:
        with open(filename,'w') as f:
            for v in tree:
                f.write('{},{}\n'.format(v[0], v[1]))

    return tree

def depth_search(graph, seed=0):
    if type(graph) != Graph:
        print('Error: graph must be of class Graph')
        exit()

    if graph.mode == 'mtx':
        print('depth_search_as_mtx not implemented yet')
        exit()
    else:
        print('depth_search_as_lst not implemented yet')
        exit()

bfs = breadth_search
dfs = depth_search


# MAIN
# -----------------------------------------------

g = Graph('exemplo.g', 'lst')
# seed = 0
# tree_filename = g.name + '_tree' + str(seed) + '.txt'
# breadth_search(g, seed, tree_filename)