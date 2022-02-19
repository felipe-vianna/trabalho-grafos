from guara.graph import Graph
from guara.paths import dijkstra, distance, mst
import time
import os
from random import randrange

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

com_peso = True
direcionado = False
tipo_de_grafo = 'lst'

qual_grafo = input("Qual grafo? (1,2,3,4 ou 5): ")
nome_arquivo_grafo = "grafo_W_{}_1.txt".format(qual_grafo)
g = Graph('./grafos/{}'.format(nome_arquivo_grafo), tipo_de_grafo, com_peso, direcionado)
origem = 0
distancias = {}
times = []
print("{}*****************  GRAFO {}  ********************{}".format(bcolors.HEADER, qual_grafo, bcolors.ENDC))

print("{}--------------- ESTUDO DE CASO 1.1  ---------------{}".format(bcolors.OKGREEN, bcolors.ENDC))
for vertice in [10,20,30,40,50]:
    destino = vertice - 1
    print("Calculando a distancia do vertice {}".format(vertice))
    distancia = distance(g, origem, destino)
    print("Distancia do vertice {}: {}".format(vertice,distancia))
    distancias[vertice] = distancia



print("{}--------------- ESTUDO DE CASO 1.2  ---------------{}".format(bcolors.OKBLUE, bcolors.ENDC))
qtd_vertices = [1000, 10000, 100000, 1000000, 5000000]
qtd_vertices_grafo = qtd_vertices[int(qual_grafo) - 1]
for k in range(100):
    print("Calculando as distancias pela {}a vez".format(k+1))
    origem = randrange(qtd_vertices_grafo)
    start_time = time.time()
    dijkstra(g, origem)
    execution_time = time.time() - start_time
    times.append(execution_time)   
print("{}Tempo medio das distancias: {}{}".format(bcolors.WARNING,sum(times)/len(times),bcolors.ENDC))



print("{}--------------- ESTUDO DE CASO 1.3  ---------------{}".format(bcolors.OKCYAN, bcolors.ENDC))
minst =  mst(g, origem, "MST-{}".format(nome_arquivo_grafo))
print("MST calculada. Calculando o peso...")
peso_mst = minst[1].sum()    
print("Peso da MST:", peso_mst)



print("{}--------------- ESTUDO DE CASO 2  ---------------{}".format(bcolors.FAIL, bcolors.ENDC))
grafo = Graph('./grafos/rede_colaboracao.txt', tipo_de_grafo, com_peso, direcionado)
origin_researcher = {2722: 'Edsger W. Dijkstra'}
target_researchers_index = { 11365: 'Alan M. Turing', 471365: 'J. B. Kruskal', 5709: 'Jon M. Kleinberg', 11386: 'Éva Tardos', 343930: 'Daniel R. Figueiredo' }
origin = 2722 - 1
print("Calculando a distancia dos pesquisadores")
distancia_pesquisadores = dijkstra(grafo, origin)
for p in target_researchers_index:
    pesquisador_nome = target_researchers_index[p]
    pesquisador = p - 1
    dist, parent = distancia_pesquisadores
    if parent[pesquisador] == -1:
        print("Nao ha caminho de Edsger Dijkstra para {}".format(pesquisador_nome))
    caminho_min = []
    v = pesquisador
    while parent[v] != v:
        caminho_min.append(v)
        v = parent[v]
    distancia_pesquisador = distancia_pesquisadores[0][pesquisador-1]
    print("Distancia do pesquisador {} até Edsger Dijkstra: {}".format(pesquisador_nome,distancia_pesquisador))
    print("Caminho minimo do pesquisador {} até Edsger Dijkstra: {}".format(pesquisador_nome,caminho_min))
    print("{}---------------{}".format(bcolors.WARNING, bcolors.ENDC))

os.system('say "thank you, next"')