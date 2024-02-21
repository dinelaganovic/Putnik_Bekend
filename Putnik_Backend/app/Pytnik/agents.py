import math, sys
from sys import maxsize

def izdvoji_zlatnike(map_content):
    zlatnici = []
    lines = map_content.strip().split('\n')
    for line in lines:
        elements = line.strip().split(',')
        brojevi = [int(element) for element in elements]
        zlatnici.append((brojevi[0],brojevi[1]))

    return zlatnici

def izdvoji_putanje(map_content):
    lines = map_content.strip().split('\n')
    broj_zlatnika = len(lines)
    matrica_putanja = [[0 for _ in range(broj_zlatnika)] for _ in range(broj_zlatnika)]
    for i, line in enumerate(lines):
        elements = line.strip().split(',')
        for j in range(2, len(elements)):
            putanja = int(elements[j].strip())
            matrica_putanja[i][j-2] = putanja
            matrica_putanja[j-2][i] = putanja
    return matrica_putanja

def broj_redova_u_stringu(map_content):
    return len(map_content.split('\n'))

def zbirPutanjaAki(zlatnici, putanje):
    poseti = set()
    trenutni = 0
    ukupna_cena = 0
    putanja = []

    while len(poseti) < len(zlatnici):
        poseti.add(trenutni)
        putanja.append(trenutni)

        sledeci = None
        min_trosak = float('inf')
        for i, trosak in enumerate(putanje[trenutni]):
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i

        if sledeci is None:
            break

        ukupna_cena += min_trosak

        trenutni = sledeci

    trosak = putanje[trenutni][0]
    ukupna_cena += trosak

    return ukupna_cena

def opisPutanjaAki(zlatnici, troskovi):
    poseti = set()
    trenutni = 0
    putanja = []
    ukupna_cena = 0
    opisi_putanja = []

    while len(poseti) < len(zlatnici):
        poseti.add(trenutni)
        putanja.append(trenutni)

        sledeci = None
        min_trosak = float('inf')
        for i, trosak in enumerate(troskovi[trenutni]):
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i

        if sledeci is None:
            break

        opis = f"{trenutni} - {sledeci}: {min_trosak}"
        opisi_putanja.append(opis)
        ukupna_cena += min_trosak

        trenutni = sledeci

    trosak_povratka = troskovi[trenutni][0]
    opis_povratka = f"{trenutni} - 0 : {trosak_povratka}"
    opisi_putanja.append(opis_povratka)
    ukupna_cena += trosak_povratka

    
    return opisi_putanja

def Aki(zlatnici, putanje):
    poseti = set()
    trenutni = 0
    putanja = []
    ukupna_cena = 0

    while len(poseti) < len(zlatnici):
        poseti.add(trenutni)
        putanja.append(trenutni)

        sledeci = None
        min_trosak = float('inf')
        for i, trosak in enumerate(putanje[trenutni]):
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i

        if sledeci is None:
            break

        ukupna_cena += min_trosak

        trenutni = sledeci

    trosak_povratka = putanje[trenutni][0]
    ukupna_cena += trosak_povratka


    putanja.append(putanja[0])
    return putanja


def next_perm(l):
    n = len(l)
    i = n-2

    while i >= 0 and l[i] > l[i+1]:
        i -= 1
    
    if i == -1:
        return False

    j = i+1
    while j < n and l[j] > l[i]:
        j += 1

    j -= 1

    l[i], l[j] = l[j], l[i]
    left = i+1
    right = n-1

    while left < right:
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return True

def opisPutanjaMat(putanje, putanja):
    opisi = []
    for i in range(len(putanja) - 1):
        start = putanja[i]
        kraj = putanja[i+1]
        daljina = putanje[start][kraj]
        opisi.append(f"{start} - {kraj}: {daljina}")
    return opisi

def Jocke(putanje):
    vertex = [i for i in range(len(putanje)) if i != 0]
    min_path = maxsize
    min_path_seq = []

    while True:
        udaljenost = 0
        k = 0
        current_seq = [0]
        for i in vertex:
            udaljenost += putanje[k][i]
            k = i
            current_seq.append(k)
        udaljenost += putanje[k][0]
        current_seq.append(0)

        if udaljenost < min_path:
            min_path = udaljenost
            min_path_seq = current_seq.copy()

        if not next_perm(vertex):
            break
    return min_path_seq

def zbirJocke(zlatnici, putanje):
    vertex = [i for i in range(len(zlatnici)) if i != 0]
    minZbir = float('inf')

    while True:
        zbirPutanja = 0
        i = 0
        for j in vertex:
            if i < len(putanje) and j < len(putanje[i]):
                zbirPutanja += putanje[i][j]
            i = j
        if i < len(putanje):
            zbirPutanja += putanje[i][0]
        minZbir = min(minZbir, zbirPutanja)

        if not next_perm(vertex):
            break
    return minZbir



maxsize = float('inf')

def copyToFinal(curr_path, final_path, N):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]

def firstMin(adj, i, N):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k] 
    return min

def secondMin(adj, i, N):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
 
        elif(adj[i][j] <= second and
             adj[i][j] != first):
            second = adj[i][j]
    return second

def TSPRec(adj, curr_bound, curr_weight, 
              level, curr_path, visited, N, final_path):
    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]]\
                                        [curr_path[0]]
            if curr_res < final_path['final_res']:
                copyToFinal(curr_path, final_path['path'], N)
                final_path['final_res'] = curr_res
        return 

    for i in range(N):
        if (adj[curr_path[level-1]][i] != 0 and not visited[i]):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)

            if curr_bound + curr_weight < final_path['final_res']:
                curr_path[level] = i
                visited[i] = True
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited, N, final_path)

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp
            visited[i] = False
 
def Uki(adj, N):
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N
    final_path = {'path': [None] * (N + 1), 'final_res': maxsize}
 
    for i in range(N):
        curr_bound += (firstMin(adj, i, N) + secondMin(adj, i, N))

    curr_bound = math.ceil(curr_bound / 2)
    visited[0] = True
    curr_path[0] = 0

    TSPRec(adj, curr_bound, 0, 1, curr_path, visited, N, final_path)
    return final_path['path'], final_path['final_res']



class TreeNode:
    def __init__(self, c_no, c_id, f_value, h_value, parent_id):
        self.c_no = c_no
        self.c_id = c_id
        self.f_value = f_value
        self.h_value = h_value
        self.parent_id = parent_id

class Tree:
    def __init__(self):
        self.nodes = {}

    def create_node(self, name, node_id, parent=None, data=None):
        new_node = TreeNode(data.c_no, data.c_id, data.f_value, data.h_value, parent)
        self.nodes[node_id] = new_node

    def get_node(self, node_id):
        return self.nodes.get(node_id)

class FringeNode:
    def __init__(self, c_no, f_value):
        self.f_value = f_value
        self.c_no = c_no

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def minKey(self, key, mstSet):
        min = sys.maxsize
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def primMST(self, g, d_temp, t):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mstSet = [False] * self.V
        parent[0] = -1

        for c in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        sum_weight = sum(self.graph[i][parent[i]] for i in range(1, self.V))
        return sum_weight


def heuristic(tree, p_id, t, V, graph):
    visited = set()
    visited.add(0)
    visited.add(t)
    if p_id != -1:
        tnode = tree.get_node(str(p_id))
        while tnode.c_id != 1:
            visited.add(tnode.c_no)
            parent_id = tnode.parent_id
            tnode = tree.get_node(str(parent_id))

    l = len(visited)
    num = V - l
    if num != 0:
        g = Graph(num)
        d_temp = {i: key for key, i in enumerate(set(range(V)) - visited)}

        for i in range(V):
            for j in range(V):
                if i not in visited and j not in visited:
                    g.graph[d_temp[i]][d_temp[j]] = graph[i][j]

        mst_weight = g.primMST(graph, d_temp, t)
        return mst_weight
    else:
        return graph[t][0]

	 
def checkPath(tree, toExpand, V):
    tnode = toExpand
    list1 = list()
    if tnode.c_id == 1:
        return 0
    else:
        depth = 0
        s = set()
        while tnode.c_id != 1:
            s.add(tnode.c_no)
            list1.append(tnode.c_no)
            parent_id = tnode.parent_id
            tnode = tree.get_node(str(parent_id))
            depth += 1
        list1.append(0)
        if depth == V and len(s) == V and list1[0] == 0:
            return 1
        else:
            return 0
		
def reconstruct_full_path_v2(tree, toExpand):
    full_path = []
    current_node_id = str(toExpand.c_id)
    while current_node_id != "1":
        current_node = tree.get_node(current_node_id)
        full_path.append(current_node.c_no)
        current_node_id = str(current_node.parent_id)
    full_path.append(0)
    return full_path[::-1]

def Micko(graph, tree, V):
    goalState = 0
    toExpand = TreeNode(0, 0, 0, 0, 0)
    key = 1
    heu = heuristic(tree, -1, 0, V, graph)
    tree.create_node("1", "1", parent=-1, data=TreeNode(0, 1, heu, heu, -1))
    fringe_list = {key: FringeNode(0, heu)}
    key += 1
    full_path = []

    while not goalState:
        minf = sys.maxsize
        for i in fringe_list.keys():
            if fringe_list[i].f_value < minf:
                current_fringe_node = fringe_list[i]
                current_tree_node_id = i
                minf = current_fringe_node.f_value

        current_tree_node = tree.get_node(str(current_tree_node_id))
        h = current_tree_node.h_value
        val = current_fringe_node.f_value - h
        path = checkPath(tree, current_tree_node, V)

        if current_tree_node.c_no == 0 and path == 1:
            goalState = 1
            cost = current_fringe_node.f_value
            full_path = reconstruct_full_path_v2(tree, current_tree_node)
        else:
            if current_tree_node_id in fringe_list:
                del fringe_list[current_tree_node_id]
            for j in range(V):
                if j != current_tree_node.c_no:
                    h = heuristic(tree, current_tree_node_id, j, V, graph)
                    f_val = val + graph[j][current_tree_node.c_no] + h
                    fringe_list[key] = FringeNode(j, f_val)
                    tree.create_node(str(current_tree_node.c_no), str(key), parent=str(current_tree_node_id),
                                     data=TreeNode(j, key, f_val, h, current_tree_node_id))
                    key += 1

    return cost, full_path
