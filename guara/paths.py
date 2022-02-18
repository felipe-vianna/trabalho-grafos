
import numpy as np
from guara.graph import Graph
from guara.utils import weighted_heap

# -----------------------------------------------
# TP1

def bfs(graph, seed, target=None, savefile=None):
    """
    Algoritmo Breadth First Search. Retorna dois vetores a partir dos quais pode-se
    remontar o caminho minimo ligando a semente a quaisquer outro vertice do grafo (bem como a 
    arvore gerada pela sua execucao) e verificar o custo desses caminhos. 
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    (int) seed:
        Semente, o vertice por onde comecar o algoritmo.
    (int) target: -opcional-
        O vertice no qual parar o calculo das distancias. Util quando deseja-se
        calcular apenas a distancia entre dois vertices.
    (str) savefile: -opcional-
        O arquivo no qual salvar a arvore gerada pela execucao do algoritmo.
    # -----------------------------------------------
    Outputs
    (int array) distance:
        O elemento distance[v] contem a distancia do vertice v ate a semente. Armazena 0 para
        a semente e infinito para os vertices desconexos dela, isto eh, para os quais nao haja
        caminho interligando-os.
    (int array) parent:
        O elemento parent[v] contem o vertice pai de v (isto eh, seu predecessor) no caminho
        ate a semente. Armazena a propria semente em distance[seed] e -1 nos vertices 
        desconexos dela, isto eh, para os quais nao haja caminho interligando-os.
    """
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    # [NOTE] Se um vertice v terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    visited = np.zeros(shape=len(graph), dtype=np.int8)
    visited[seed] = True # iniciamos a descoberta dos vertices pela semente

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    distance = np.full(shape=len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1
    distance[seed] = 0 # definimos o nivel da raiz como 0

    # parent[v]: o vertice pai de v na arvore gerada pela busca
    parent = np.full(shape=len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # queue: fila dos vertices que precisam ser investigados
    queue = [seed]

    while (len(queue) != 0):
        v = queue[0] # pegamos o vertice do inicio da fila
        queue.pop(0) # retiramos ele da fila

        for n in graph.neighbors(v):
            if n != v: # para evitar loops, ignoramos arestas do tipo (i,i)
                if not visited[n]: # o vizinho ainda nao foi investigado
                    queue.append(n) # adicionamos o vizinho na fila para ser investigado
                    visited[n] = True # indicamos que o vizinho ja foi descoberto

                    distance[n] = distance[v] + 1 # declaramos o vizinho como estando 1 nivel acima do atual
                    parent[n] = v # declaramos o vertice atual v como o pai de n na arvore

                    if n == target:
                        queue = []
                        break # quando encontramos o vertice de destino, saimos do loop

    if savefile:
        with open(savefile,'w') as f:
            for v in graph:
                f.write('{},{}\n'.format(parent[v], distance[v]))

    return distance, parent

def dfs(graph, seed, savefile=None):
    """
    Algoritmo Depth First Search. Retorna dois vetores a partir dos quais pode-se
    remontar o caminho minimo ligando a semente a quaisquer outro vertice do grafo (bem como a 
    arvore gerada pela sua execucao) e verificar o custo desses caminhos. 
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    (int) seed:
        Semente, o vertice por onde comecar o algoritmo.
    (str) savefile: -opcional-
        O arquivo no qual salvar a arvore gerada pela execucao do algoritmo.
    # -----------------------------------------------
    Outputs
    (int array) distance:
        O elemento distance[v] contem a distancia do vertice v ate a semente. Armazena 0 para
        a semente e -1 para os vertices desconexos dela, isto eh, para os quais nao haja
        caminho interligando-os.
    (int array) parent:
        O elemento parent[v] contem o vertice pai de v (isto eh, seu predecessor) no caminho
        ate a semente. Armazena a propria semente em distance[seed] e -1 nos vertices 
        desconexos dela, isto eh, para os quais nao haja caminho interligando-os.
    """
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    # [NOTE] Se um vertice v terminar com distance[v] == -1 e/ou parent[v] == -1, significa que ele nao
    #    esta conectado a raiz

    # visited[v]: flag booleana indicando se o vertice v ja foi investigado
    visited = np.zeros(shape=len(graph), dtype=np.uint8) # flags indicando se o vertice ja foi visitado

    # distance[v]: representa a distancia, em arestas, de v ate a raiz, eh equivalente ao nivel do
    #    vertice na arvore gerada pela busca
    distance = np.full(shape=len(graph), fill_value=(-1), dtype=np.int16) # array de tamanho n preenchido com -1
    distance[seed] = 0 # definimos o nivel da raiz como 0

    # parent[v]: o vertice pai de v na arvore gerada pela busca
    parent = np.full(shape=len(graph), fill_value=(-1), dtype=np.int32) # array de tamanho n preenchido com -1
    parent[seed] = seed # colocamos a raiz como pai de si mesma

    # stack: pilha dos vertices que precisam ser investigados
    stack = [seed]

    while (len(stack) != 0):
        v = stack[-1] # pegamos o vertice do topo da pilha
        stack.pop(-1) # retiramos ele da pilha
        
        if not visited[v]:
            visited[v] = True

            # [REF] https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
            for n in graph.neighbors(v)[::-1]:
                # seguindo o padrao da aula 5 (slide 18) percorremos os vizinhos em ordem descrescente
                # assim os vizinhos de menor indice ficam no topo da pilha para serem investigados primeiro
                if not visited[n]: # se o vizinho ja foi visitado, nao ha pq adiciona-lo na pilha
                    stack.append(n)

                    distance[n] = distance[v] + 1 # o nivel do noh neighbor eh um nivel acima do atual
                    parent[n] = v # o pai do vertice n eh o vertice v sendo analisado

    if savefile:
        with open(savefile,'w') as f:
            for v in graph:
                f.write('{},{}\n'.format(parent[v], distance[v]))

    return distance, parent

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

def diameter(graph):
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    diameter = 0

    for v in graph:
            distances = bfs(graph, seed=v)[0] # pegando os tamanhos de todos os caminhos simples que saem de v
            longest = distances.max() # pegando o tamanho do maior caminho simples que sai de v

            if longest > diameter:
                diameter = longest

    return diameter   


# -----------------------------------------------
# TP2

def distance(graph, seed, target):
    """
    Calcula a distancia entre dois vertices em um grafo. Para grafos nao-direcionados e
    sem pesos, eh executado algoritmo de BFS. Para nao-direcionados com pesos nao-negativos,
    executa-se o algoritmo de Dijkstra. Em grafos direcionados, executa-se o algoritmo de
    Bellman-Ford.
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    (int) seed:
        Semente, o vertice de origem, por onde comecar o algoritmo.
    (int) target:
        O vertice de destino, no qual parar o calculo das distancias.
    # -----------------------------------------------
    Outputs
    (int | float | None) distance:
        Distancia calculada. Quando o grafo eh nao-direcionado e sem pesos, eh retornado um inteiro,
         que sera igual a -1 caso nao haja nenhum caminho ligando os vertices. Quando o grafo eh
         nao-direcionado e com pesos negativos, eh retornado None. Nos demais casos, sera um numero
         em ponto flutuante, igual a infinito caso nao haja nenhum caminho entre os vertices.
    """
    if type(graph) != Graph:
        raise TypeError('graph must be instance of class Graph')

    if not graph.directed:
        if not graph.weighted:
            return bfs(graph, seed, target)[0][target]

        elif not graph.has_neg_weights:
            return dijkstra(graph, seed, target)[0][target]

    else:
        return bellman_ford(graph, target)[0][seed]

    return None

# -----------------------------------------------


def dijkstra(graph, seed, target=None):
    """
    Executa o algoritmo de Dijkstra. Retorna dois vetores a partir dos quais pode-se
    remontar o caminho minimo ligando a semente a quaisquer outro vertice do grafo (bem como a 
    arvore gerada pela execucao) e verificar o custo desses caminhos.
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    (int) seed:
        Semente, o vertice por onde comecar o algoritmo.
    (int) target:
        O vertice de destino, no qual parar o calculo das distancias. Util quando deseja-se
        calcular apenas a distancia entre dois vertices.
    # -----------------------------------------------
    Outputs
    (float array) distance:
        O elemento distance[v] contem a distancia do vertice v ate a semente. Armazena 0 para
        a semente e infinito para os vertices desconexos dela, isto eh, para os quais nao haja
        caminho interligando-os.
    (int array) parent:
        O elemento parent[v] contem o vertice pai de v (isto eh, seu predecessor) no caminho
        ate a semente. Armazena a propria semente em distance[seed] e -1 nos vertices 
        desconexos dela, isto eh, para os quais nao haja caminho interligando-os.
    """

    # dist = [ np.inf for v in range(len(graph)) ]
    dist = np.full(shape=len(graph), fill_value=np.inf)
    dist[seed] = 0

    # parent = [ -1 for v in range(len(graph)) ]
    parent = np.full(shape=len(graph), fill_value=(-1))
    parent[seed] = seed

    # O heap ponderado, onde cada elemento tem uma chave identificadora e o peso que define 
    # sua ordem de prioridade. Usaremos os indices dos vertices como as chaves e as distancias 
    # ate a semenete como seus pesos
    heap = weighted_heap()

    for v in graph:
        heap.push(v, dist[v])

    v = heap.pop()
    while (v is not None) and (v != target):
        v = v[0] # lembrando que cada elemento do heap consiste de (chave, peso) pegamos a chave
        for n in graph.neighbors(v):
            e = graph.edge(v,n)

            if dist[n] > dist[v] + e:
                parent[n] = v
                dist[n] = dist[v] + e
                heap.update(n, dist[n])

        v = heap.pop()

    return dist, parent

# -----------------------------------------------

def floyd_warshall(graph):
    """
    Executa o algoritmo de Floyd-Warshall. Retorna duas matrizes (arrays de 2-dim) a partir 
    das quais pode-se remontar o caminho minimo entre quaisquer vertices do grafo e verificar o
    custo desses caminhos.
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    # -----------------------------------------------
    Outputs
    (float array) distance:
        O elemento dist[i,j] contem o custo total do menor caminho que se conhece saindo do vertice i
        ate o vertice j. Armazena 0 em dist[i,i] e infinito para os vertices i,j desconexos, isto eh,
        para os quais nao haja caminho interligando-os.
    (int array) parent:
        O elemento parent[i,j] contem o vertice pai de j (isto eh, seu predecessor) no caminho saindo
        do vertice i ate o vertice j. Armazena i em dist[i,i] e -1 para os vertices i,j desconexos, 
        isto eh, para os quais nao haja caminho interligando-os.
    (bool) neg_cycle:
        Flag que indica se foi identificado um ciclo negativo no grafo. Caso tenha sido, a distancia
        entre os vertices no grafo nao eh bem-definida e os valores armazenados nas matrizes nao sao
        necessariamente representativos das distancias e dos caminhos minimos no grafo.
    """

    # dist[i,j]: contem o custo total do menor caminho que se conhece que vai de i para j
    dist = np.full( shape=(len(graph), len(graph)),
                    fill_value=np.inf )

    neg_cycle = False # flag indicando se foi encontrado um ciclo negativo no grafo

    # inicializamos dist com o custo para ir de um vertice a outro sem vertices intermediarios
    for v in graph:
        dist[v][ graph.neighbors(v) ] = graph.weights(v) # selecionamos, na linha v, os elementos correpospondentes aos vizinhos de v
        dist[v,v] = 0 # o custo de ir de v para v eh nulo

    # parent[i,j]: contem o vertice anterior a j no menor caminho que se conhece de i para j (isto eh, o pai de j)
    parent = np.full( shape=(len(graph), len(graph)),
                      fill_value=(-1) )

    for v in graph:
        parent[v][ graph.neighbors(v) ] = v
        parent[v,v] = v

    for k in range(1, len(graph)):
        # comecamos o k em 1, pois k = 0 eh a propria inicializacao da matriz
        # print(f'k:{k-1}')
        # print(dist)
        # print(parent)
        for i in graph:
            for j in graph:
                if (dist[i,k] + dist[k,j]) < dist[i,j]:

                    dist[i,j] = (dist[i,k] + dist[k,j])
                    parent[i,j] = parent[k,j]

                    if (i == j) and (dist[i,i] < 0):
                        neg_cycle = True
                        # print(f'i:{i}, j:{j}, k:{k}')
                        # print(dist)
                        # print(parent)
                        # raise Exception(f'Encontrado ciclo negativo no grafo {graph.name}. Nao foi possivel calcular distancias.')

    return dist, parent, neg_cycle

# -----------------------------------------------


def bellman_ford(graph, target):
    """
    Executa o algoritmo de Bellman-Ford. Retorna dois vetores a partir dos quais pode-se remontar
    o caminho minimo que vai de quaisquer vertices do grafo para o de destino e verificar
    o custo desse caminho.
    # -----------------------------------------------
    Inputs
    (class Graph) graph:
        Grafo no qual executar o algoritmo.
    (int) target:
        O destino, vertice final de todos os caminhos para os quais sao calculados os custos.
    # -----------------------------------------------
    Outputs
    (float array) distance:
        O elemento dist[v] contem o custo total do menor caminho que se conhece saindo do vertice v
        ate o vertice destino. Armazena 0 em dist[target] e infinito para os vertices desconexos
        do destino, isto eh, para os quais nao haja caminho interligando-os.
    (int array) after:
        O elemento after[v] contem o vertice seguinte a v no caminho saindo
        de v ate o vertice destino. Armazena o proprio destino em after[target] e -1 nos
        vertices desconexos dele, isto eh, para os quais nao haja caminho interligando-os.
    (bool) neg_cycle:
        Flag que indica se foi identificado um ciclo negativo no grafo. Caso tenha sido, a distancia
        entre os vertices no grafo nao eh bem-definida e os valores armazenados nos vetores nao sao
        necessariamente representativos das distancias e dos caminhos minimos no grafo.
    """
    dist = np.full(shape=len(graph), fill_value=np.inf)
    dist[target] = 0

    after = np.full(shape=len(graph), fill_value=(-1))
    after[target] = target

    # update[v]: flag para indicar se devemos (tentar) atualizar as distancias do vertice v
    update = np.full(shape=len(graph), fill_value=True)
    update[target] = True

    stop = False
    for i in range(len(graph)-1):
        if stop:
            break # interrompemos o loop
        else:
            stop = True

        for v in graph:
            if update[v]:
                update[v] = False
                for n in graph.neighbors(v):
                    e = graph.edge(v,n)
                    if (dist[v] > dist[n] + e):
                        dist[v] = dist[n] + e
                        after[v] = n
                        update[ graph.neighbors(v) ] = True # no proximo loop tentaremos atualizar as distancias de todos os vizinhos de v
                        stop = False # se houve atualizacao em algum vertice, continuamos o algoritmo

    # Ao final, executamos mais um loop para verificar se ha atualizacao em alguma distancia
    # pois, se houver, eh porque o grafo possui um ciclo negativo
    neg_cycle = False
    for v in graph:
        if neg_cycle: # ja encontramos um ciclo negativo, podemos encerrar a busca
            break
        for n in graph.neighbors(v):
            e = graph.edge(v,n)
            if (dist[v] > (dist[n] + e)):
                neg_cycle = True
                break # ja encontramos um ciclo negativo, podemos encerrar a busca

    return dist, after, neg_cycle

# -----------------------------------------------


def mst(graph, seed, savefile=None):
    """
    Recebe um  grafo nao-direcionado e um vertice semente para o algoritmo de Prim, retornando
    a MST resultante.

    A MST eh retornada como uma tupla (parent, cost), onde parent e cost sao listas no formato:

        parent[v]: o vert 'u' da aresta (u,v) presente na MST (para v != seed)
        cost[v]: o peso da aresta (u,v) descrita por parent[v] (para v != seed)
        parent[seed]: seed
        cost[seed]: 0

    Para obter o custo total da arvore, pode-se usar o metodo cost.sum()

    Caso o grafo seja desconexo, nao existe (por definicao) uma MST. Ainda assim, os vetores
    resultantes da execucao do algoritmo sao retornados. O usuario pode identificar esse caso
    quando parent[v] == -1 para algum v, pois isso indica que esse vertice nao foi inserido
    na arvore.

    Caso seja especificado um arquivo em savefile, cada linha no arquivo contem
    as arestas (u,v) da arvore na forma:

        "u v"

    Ou seja, apenas os vertices separados por um espaco em branco.
    """

    # cost = [ np.inf for v in graph ]
    cost = np.full(shape=len(graph), fill_value=np.inf)
    cost[seed] = 0

    # parent = [ -1 for v in graph ]
    parent = np.full(shape=len(graph), fill_value=(-1)) # nesse algoritmo, o pai de um vertice eh o no que o inseriu na arvore
    parent[seed] = seed

    # O heap ponderado, onde cada elemento tem uma chave identificadora e o peso que define 
    # sua ordem de prioridade. Usaremos os indices dos vertices como as chaves e as distancias 
    # ate a semenete como seus pesos
    heap = weighted_heap()

    for v in graph:
        heap.push(v, cost[v])

    v = heap.pop() # tiramos a semente do heap

    while v is not None:
        v = v[0] #lembrando que cada elemento no heap contem chave e peso, pegamos a chave

        for n in graph.neighbors(v):
            e = graph.edge(v,n)

            if (cost[n] > e) and (n in heap):
                # Ao percorrermos os vizinhos de v, avaliamos apenas os que ainda nao
                # estiverem na arvore
                parent[n] = v
                cost[n] = e
                heap.update(n, cost[n])

        v = heap.pop()

    # if (-1 in parent): 
    #     # ha componentes desconexas no grafo, entao nao existe MST desse grafo
    #     # (por definicao a MST conecta todos os vertices do grafo)
    #     return None, None

    if savefile:
        with open(savefile,'w') as f:
            # f.write(f'{len(cost)} {cost.sum()}')
            for v in graph:
                if v != seed:
                    # f.write(f'{parent[v]} {v} {cost[v]}\n')
                    f.write(f'{parent[v]} {v}\n')

    return parent, cost