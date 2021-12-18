import os
import numpy as np

class Graph:
    def __init__(self, filename, memory_mode='mtx'):

        # Verificando se o nome do arquivo passado eh uma string
        if type(filename) != type(''):
            print('Error: Graph expects a string with the path to the .txt that describes the graph.')
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
            print('read_as_list() not implemented yet')
            print('Exiting...')
            exit()

    # } __init__

    def read_as_matrix(self,filename):

        # Lendo o arquivo
        lines = []
        with open(filename) as f:
            lines = f.readlines() # lista de strings contendo o conteudo de cada linha

        self.__shape['v'] = int(lines[0]) # numero de vertices

        self.__graph = np.zeros((self.__shape['v'],self.__shape['v']), dtype=np.uint8) # inicializando a matriz com zeros (False)
        self.__degrees = np.zeros(self.__shape['v'], dtype=np.uint) # inicializando todos os graus dos vertices em zero

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
        if not self.__dg_min:
            self.__dg_min = self.__degrees.min()
        return self.__dg_min
    
    @property
    def dg_max(self):
        if not self.__dg_max:
            self.__dg_max = self.__degrees.max()
        return self.__dg_max

    @property
    def dg_avg(self):
        if not self.__dg_avg:
            self.__dg_avg = self.__degrees.mean()
        return self.__dg_avg

    @property
    def dg_median(self):
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



# MAIN
# -----------------------------------------------

g = Graph('exemplo.txt')