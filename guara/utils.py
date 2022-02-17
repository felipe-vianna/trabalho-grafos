

class weighted_heap():
    # [REF] https://docs.python.org/3/library/heapq.html
    # a implementacao de heap que precisavamos era um pouco diferente da nativa, mas ela serve como refer
    def __init__(self):
        self.queue = [] # a fila de prioridade
        # [TODO] implementar tabela de acesso a fila para otimizar a atualizacao de um elemento
        self.lookup = {} # tabela de acesso de elementos do heap por meio da chave

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, key):
        return self.queue[ self.lookup[key] ]

    # -------------------------------------------------------------------

    def push(self, key, weight):
        """
        Insere um elemento dado por uma chave (identificador) e um peso. Reordena os elementos da
        fila com base nos seus pesos.
        Complexidade: O(log(n))
        """

        self.lookup[key] = len(self.queue)
        self.queue.append((key, weight)) # adicionamos o elemento no final do heap

        print(self.queue)

        self.climb(key) # subimos o elemento para o local correto no heap

    # -------------------------------------------------------------------

    def pop(self):
        """
        Tira o primeiro elemento da fila e a reordena. Retorna o primeiro elemento.
        Complexidade: O(log(n))
        """
        if len(self.queue) == 0:
            return None


        if len(self.queue) > 1:
            self.switch(self.queue[0], self.queue[-1])

        # lembrando que o ultimo e o primeiro elemento foram trocados
        last = self.queue[0]
        first = self.queue.pop(-1)
        del self.lookup[ first[0] ]

        print(self.queue)

        self.fall(key=last[0]) # descemos o elemento que foi colocado no topo para o local correto

        return first

    # -------------------------------------------------------------------

    def update(self, key, new_weight):
        """
        Atualiza o peso de um elemento, identificado pela chave, e reordena o heap.
        Complexidade: O(log(n))
        """
        if len(self.queue) == 0:
            return None

        idx = self.lookup[key]
        self.queue[idx] = (key, new_weight)

        # subimos (ou descemos, o que for possivel) o elemento modificado ate o heap estar ordenado
        self.fall(key)
        self.climb(key)
    
    # -------------------------------------------------------------------

    def climb(self, key):
        """
        Recebe um indice de elemento na fila e "sobe" esse elemento (troca ele com seus pais)
        ate que a ordem correta do heap seja alcancada.
        """
        if len(self.queue) == 0:
            return None

        elem = self[key]
        parent = self.parent(key) # o pai do elemento que esta subindo

        while (parent is not None) and (parent[1] > elem[1]):
            # se o peso do pai for maior que do elemento atual, trocamos a posicao deles na fila
            self.switch(elem, parent)

            print(self.queue)

            parent = self.parent(key)

        return None

    # -------------------------------------------------------------------

    def fall(self, key):
        """
        Recebe um indice na fila de um elemento e faz esse elemento "cair" (troca-o de lugar com o
        menor dos seus filhos) ate que o heap esteja ordenado.
        """
        if len(self.queue) == 0:
            return None

        elem = self[key]
        child = self.smaller_child(key) # o menor filho do elemento que esta descendo

        while (child is not None) and (child[1] < elem[1]):
            # se o peso do filho for menor que do elemento atual, trocamos a posicao deles na fila
            self.switch(elem, child)

            print(self.queue)

            child = self.smaller_child(key)

        return None

    # -------------------------------------------------------------------

    def parent(self, key):
        """
        Recebe a chave de um elemento da fila e retorna o pai dele.
        """
        # elem_index = self.lookup[key]
        parent_index = (self.lookup[key] - 1)//2 # o indice do pai do elemento

        if parent_index < 0:
            # se o indice do pai na fila for negativo, eh pq ele nao existe e nao ha como subir
            return None

        return self.queue[parent_index]

    # -------------------------------------------------------------------

    def smaller_child(self, key):
        """
        Recebe a chave de um elemento da fila e retorna o filho dele que possuir o menor peso.
        """
        # elem_index = self.lookup[key]
        child_index = (2 * self.lookup[key]) + 1 # o indice do primeiro filho (caso exista)

        if child_index >= len(self.queue): # verificamos se o filho existe
            return None # o filho nao existe


        if child_index < (len(self.queue) - 1): # verificamos se o segundo filho existe

            child_index_2 = child_index + 1 # o indice do segundo filho (caso exista)

            # pegaremos o filho com menor peso
            if (self.queue[child_index_2][1] < self.queue[child_index][1]):
                child_index = child_index_2

        return self.queue[child_index]

    # -------------------------------------------------------------------
    
    def switch(self, elem1, elem2):
        """
        Recebe dois elementos (tupla) do heap e troca eles de lugar, atualizando a tabela 
        de acesso de acordo.
        """

        idx1 = self.lookup[ elem1[0] ]
        idx2 = self.lookup[ elem2[0] ]

        # trocamos a posicao dos elementos na fila
        self.queue[idx1] = elem2
        self.queue[idx2] = elem1

        # trocamos os indices na tabela de acesso
        self.lookup[ elem1[0] ] = idx2
        self.lookup[ elem2[0] ] = idx1
