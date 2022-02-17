from guara.graph import Graph
from guara.paths import distance, mst
import time
import os

com_peso = True if input("O grafo possui peso? (0 para não, 1 para sim): ") == '1' else False
direcionado = True if input("O grafo eh direcionado? (0 para não, 1 para sim): ") == '1' else False
tipo_de_grafo = 'lst'
g = Graph('./guara/exemplo.g', tipo_de_grafo, com_peso, direcionado)
origem = 1
distancias = {}
times = []
for vertice in [10,20,30,40,50]:
    start_time = time.time()
    destino = vertice
    print("Calculando a distancia do vertice {}".format(destino))
    distancia = distance(g, origem, destino)
    print("Distancia do vertice {}: {}".format(destino,distancia))
    distancias[destino] = distancia
    execution_time = time.time() - start_time
    times.append(execution_time)
print("Distancia de cada conjunto de vertices:", distancias)
print("Tempo medio das distancias: {}".format(sum(times)/len(times)))
minst =  mst(g, origem, 'saidaTP2.txt')
peso_mst = minst[1].sum()    
print("Peso da MST:", peso_mst)
os.system('say "thank you, next"')