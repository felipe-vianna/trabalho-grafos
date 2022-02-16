

class weighted_heap():
    # [REF] https://docs.python.org/3/library/heapq.html
    # a implementacao de heap que precisavamos era um pouco diferente da nativa, mas ela serve como refer
    def __init__(self):
        self.queue = [] # a fila de prioridade

    def __len__(self):
        return len(self.queue)

    # -------------------------------------------------------------------

    def insert(self, key, weight):
        """
        Insere um elemento dado por uma chave e um peso. A chave eh o identificador do elemento, que
        eh retornada quando retira-se o elemento do heap. O peso eh o valor utilizado para determinar
        a prioridade, a posicao do elemento na fila.
        """

        self.queue.append((key, weight)) # adicionamos o elemento no final do heap

        # print(self.queue)

        self.climb(index=(len(self.queue) - 1)) # subimos o elemento para o local correto no heap

    # -------------------------------------------------------------------

    def remove(self):
        """
        Tira o primeiro elemento da fila e a reordena. Retorna o primeiro elemento.
        """

        if len(self.queue) == 0:
            return None

        root = self.queue[0] # o primeiro elemento do heap

        if len(self.queue) > 1:
            self.queue[0] = self.queue.pop(-1) # retiramos o ultimo elemento e colocamos no topo do heap
        else:
            self.queue.pop(-1) # retiramos o ultimo elemento do heap

        # print(self.queue)

        self.fall(index=0) # descemos o elemento que foi colocado no topo para o local correto

        return root

    # -------------------------------------------------------------------

    def update(self, key, new_weight):

        queue_index = 0
        for idx, elem in enumerate(self.queue):
            if elem[0] == key:
                queue_index = idx
                self.queue[idx] = (key, new_weight)
                break

        # print(self.queue)

        # subimos (ou descemos, o que for possivel) o elemento modificado ate o heap estar ordenado
        self.fall(queue_index)
        self.climb(queue_index)
    
    # -------------------------------------------------------------------

    def climb(self, index):
        """
        Recebe um indice de elemento na fila e "sobe" esse elemento (troca ele com seus pais)
        ate que a ordem correta do heap seja alcancada
        """
        parent_index = (index - 1)//2 # o indice na fila do pai do elemento que esta subindo

        if parent_index < 0:
            # se o indice do pai na fila for negativo, eh pq ele nao existe e nao ha como subir
            return None

        while self.queue[parent_index][1] > self.queue[index][1]:
            # se o peso do pai for maior que o elemento atual, trocamos a posicao deles na fila
            temp = self.queue[parent_index]
            self.queue[parent_index] = self.queue[index]
            self.queue[index] = temp

            # print(self.queue)

            # atualizamos o indice do elemento que esta subindo e do seu pai
            index = parent_index
            parent_index = (index - 1)//2

            if parent_index < 0:
                # se o indice do pai na fila for negativo, nao ha mais como subir
                return None

    # -------------------------------------------------------------------

    def fall(self, index):
        """
        Recebe um indice na fila de um elemento e faz esse elemento "cair" (troca-o de lugar com o
        menor dos seus filhos) ate que o heap esteja ordenado
        """

        child_index = self.get_smaller_child(index)

        if child_index is None:
            return None

        # -----------------------------------
        while (self.queue[index][1] > self.queue[child_index][1]):
            # trocamos a posicao dos elementos na fila
            temp = self.queue[index]
            self.queue[index] = self.queue[child_index]
            self.queue[child_index] = temp

            # print(self.queue)

            index = child_index

            child_index = self.get_smaller_child(index)

            if child_index is None:
                return None

    # -------------------------------------------------------------------

    def get_smaller_child(self, index):
        """
        Recebe o indice de um elemento na fila e retorna o indice do filho que possuir o menor peso
        """
        child_index = (2 * index) + 1 # o indice do primeiro filho (caso exista)

        if child_index >= len(self.queue): # verificamos se o filho existe
            return None

        child_index_2 = child_index + 1 # o indice do segundo filho (caso exista)

        if child_index_2 < len(self.queue): # verificamos se o segundo filho existe

            # pegaremos o filho com menor peso
            if (self.queue[child_index_2][1] < self.queue[child_index][1]):
                child_index = child_index_2

        return child_index