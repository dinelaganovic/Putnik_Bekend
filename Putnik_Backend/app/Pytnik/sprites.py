import math, sys
from sys import maxsize

#informacije o zlatnicima, zlatnici su koordinate
def izdvoji_zlatnike(map_content):
    zlatnici = []
    lines = map_content.strip().split('\n')
    for line in lines:
        elements = line.strip().split(',')
        brojevi = [int(element) for element in elements]
        zlatnici.append((brojevi[0],brojevi[1]))
    return zlatnici

#matrica putanja izmedju razlicitih čvorova(zlatnika)
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

#izbroji redove
def broj_redova_u_stringu(map_content):
    return len(map_content.split('\n'))

#ova funkcija pronalazi putanju koja obilazi sve cvorove i vraca seu  pocetni cvor i racuna ukupnu vrednost te putanje
def zbirPutanjaAki(zlatnici, putanje):
    poseti = set()
    trenutni = 0
    ukupna_cena = 0
    putanja = []    #cuva redosled posecenih cvorova
    while len(poseti) < len(zlatnici):
        poseti.add(trenutni)
        putanja.append(trenutni)
        sledeci = None
        min_trosak = float('inf')
            #prolazimo kroz sve grane iz trenutnog cvora i trazimo najmanju granu do cvora koji nije posecen
            #enumerate vraca u svakoj iteraciji 2 elementa, to je za odabir sledeceg cvora za posetu
            #a trosak je putanja od trenutnog cvora do cvora i
        for i, trosak in enumerate(putanje[trenutni]):
            #proveravamo da li je i u skupu poseti
            #uporedjuje se vrednost putanje od trenutnog cvora do sledeceg i sa min_trosak.
            #ako pronadjemo manji, azurira se mintrosak, a 'sledeci' postaje index tog cvora
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i
            #ako ne postoji nijedan cvor koji nije posecen
        if sledeci is None:
            break

        ukupna_cena += min_trosak

        trenutni = sledeci     #dodajemo vrednost povratka u pocetni cvor
    trosak = putanje[trenutni][0]
    ukupna_cena += trosak
    return ukupna_cena

#belezimo opis svake pojedinacne putanje koju algoritam bira
#postavljamo cvorove i matricu izmedju cvorova

def opisPutanjaAki(zlatnici, troskovi):
    poseti = set()
    trenutni = 0
    putanja = []    #lista koja čuva redosled posecenih
    ukupna_cena = 0
    opisi_putanja = []    #lista koja čuva opis svake putanje

    #poseta svi cvorova
    while len(poseti) < len(zlatnici):
        poseti.add(trenutni)
        putanja.append(trenutni)

        sledeci = None
        min_trosak = float('inf')        #najmanji trosak do sledeceg cvora
        for i, trosak in enumerate(troskovi[trenutni]): #posecujemo cvorove dostupne iz trenutnog cvora i biramo sa najmanjom putanjom koja nije posecena
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i

        if sledeci is None:         #zavrsavamo ako nema vise cvorova da posecivanje
            break

        opis = f"{trenutni} - {sledeci}: {min_trosak}" #opis putanje i dodajemo u opisiputanje i trenutnu duzinu u ukupni trosak
        opisi_putanja.append(opis)
        ukupna_cena += min_trosak

        trenutni = sledeci
    #ovde racunamo i dodajemo u pocetni cvor
    trosak_povratka = troskovi[trenutni][0]
    opis_povratka = f"{trenutni} - 0 : {trosak_povratka}"
    opisi_putanja.append(opis_povratka)
    ukupna_cena += trosak_povratka

    
    return opisi_putanja

def Aki(zlatnici, putanje): # pretraga po dubini
    poseti = set()
    trenutni = 0
    putanja = [] #lista koja cuva redoslec posecenih cvorova
    ukupna_cena = 0

    while len(poseti) < len(zlatnici):
        #dodajemo trenutni cvor u skup posecenih
        poseti.add(trenutni)
        putanja.append(trenutni)

        sledeci = None
        min_trosak = float('inf')
        for i, trosak in enumerate(putanje[trenutni]): #enumerate vraca elemente
            if trosak < min_trosak and i not in poseti:
                min_trosak = trosak
                sledeci = i

        if sledeci is None:
            break
    #dodajemo minimalni put do sledeceg cvora i cuvamo vrednost
        ukupna_cena += min_trosak

        trenutni = sledeci

    trosak_povratka = putanje[trenutni][0]
    ukupna_cena += trosak_povratka


    putanja.append(putanja[0])
    return putanja


#od jedne mogucnosti pravi drugu, npr 123,132..ide dalje dok ne odradi sve 
#kad odradi vraca putanju
def next_perm(l):
    n = len(l) #odredjujemo duzinu
    i = n-2 #pocetni index i postavljamo na pretposlednji
    #petlja se krece obrnutim redom, trazimo prvi element l[i] koji je manji od sledeceg
    #prekretnica u mijenjanju
    while i >= 0 and l[i] > l[i+1]:
        i -= 1
    
    if i == -1:
        return False

    j = i+1
    while j < n and l[j] > l[i]:#kada pronadjemo l[i], potrebno je pronaci najmanji elem koji je veci od l[i]
        j += 1

    j -= 1
    #ovde se vrsi zamena mesta i j, formira se desno u rastucem
    l[i], l[j] = l[j], l[i]
    left = i+1
    right = n-1

    while left < right:
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return True

#opisi putanje izmedju cvorova, putanje je matrica a putanja je niz koji predstavlja putanje koje treba opisati
def opisPutanjaMat(putanje, putanja):
    opisi = []#lista za cuvanje opisa putanja
    for i in range(len(putanja) - 1):#start je trenutni cvor a kraj je sledeci cvor u putanji
        start = putanja[i]
        kraj = putanja[i+1]
        daljina = putanje[start][kraj]
        opisi.append(f"{start} - {kraj}: {daljina}")
    return opisi

def Jocke(putanje):

    vertex = [i for i in range(len(putanje)) if i != 0] #lista svih cvorova bez poslednjeg
    min_path = maxsize #max vrednost za praznu putanju
    min_path_seq = []    #cuvanje najkrace putanje

    while True:
        udaljenost = 0
        k = 0
        current_seq = [0]
        for i in vertex:
            udaljenost += putanje[k][i]  #azuriramo trenutni cvor
            k = i
            current_seq.append(k)#dodajemo udaljenost od poslednjeg cvora do pocetnog

        udaljenost += putanje[k][0]
        current_seq.append(0)
    
    #ako je trenutna putanja kraca od najkrace, azuriraj najkracu putanju
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
        i = 0#pocetni cvor
        for j in vertex:
        #idemo kroz listu cvorova osim pocetnog i racunamo ukupan zbir, azurira trenutni cvor posle svake iteracije
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

def copyToFinal(curr_path, final_path, N):#curr_path trenutni put koji se istrazuje, i krajni put
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]
    #nalazimo najmanju duzinu grane
    #adj matrica susedtsva, i je trenutni cvor a N broj cvorova
def firstMin(adj, i, N):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k] 
    return min

def secondMin(adj, i, N):#druga najmanja tezina koja izlazi i odredjenog cvora
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
#branch bount resenje
def TSPRec(adj, curr_bound, curr_weight, 
              level, curr_path, visited, N, final_path):
    #curr_weigth trenutna ukupna duzina puta 
    #curr_path je niz koji pravi trenutni put, a level je broj koji je posecen, a N je ukupan broj 
    #proveravamo da li smo dostigli koji odgovara broju cvorova 
    if level == N:#proveravamo da li postoji putanja koja povezuje poslednji cvor sa pocetnim
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            #ako postoji, da izracuna ukupan put 
            curr_res = curr_weight + adj[curr_path[level - 1]]\
                                        [curr_path[0]]
           #provera da li je trenutni put manji od najbolje pronadjene putaanje
            if curr_res < final_path['final_res']:
                #ako jeste, kopiraj trenutni put u krajni rezultat
                copyToFinal(curr_path, final_path['path'], N)
                final_path['final_res'] = curr_res
        return 

    for i in range(N): 
        #proverava da li postoji veza izmedju poslednjeg cvora u putu i cvora i, i da li cvor i nije posecen
        if (adj[curr_path[level-1]][i] != 0 and not visited[i]):
            temp = curr_bound#cuva trenutnu procenu duzine puta pre nego sto pristupi izmenama

            curr_weight += adj[curr_path[level - 1]][i]
            #drugi nivo
            if level == 1:
                #proverava duzinu puta na osnovu najmanje
                curr_bound -= ((firstMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)

            if curr_bound + curr_weight < final_path['final_res']:
                curr_path[level] = i
                visited[i] = True
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited, N, final_path)

            curr_weight -= adj[curr_path[level - 1]][i]
             #ako je uslov ispunjen, cvor i se dodaje u trenutni put i oznacava kao posecen
            curr_bound = temp
            visited[i] = False
#grananje i ogranicavanje branch and bount search
#udaljenosti i broj grafova
def Uki(adj, N):
    #pracenje gornje granice za duzinu puta
    curr_bound = 0 #curr_bound duzina puta
    #svaka pozicija u nizu predstavlja trenutnu putanju, i vracamo je na pocetak
    curr_path = [-1] * (N + 1)
    visited = [False] * N#proveravamo da li smo posetili ili ne 
    final_path = {'path': [None] * (N + 1), 'final_res': maxsize}
    #prolazimo kroz svaki cvor, racunamo prvu i drugu najmanju udaljenost 
    for i in range(N):
        curr_bound += (firstMin(adj, i, N) + secondMin(adj, i, N))
    #uzimamo polovinu trentne vrednosti
    curr_bound = math.ceil(curr_bound / 2)
    visited[0] = True
    curr_path[0] = 0
    #rekurzivna funkcija, za posecen samo jednom i vraca na pocetak
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited, N, final_path)
    return final_path['path'], final_path['final_res']



class TreeNode:
    def __init__(self, c_no, c_id, f_value, h_value, parent_id):# cvor, c_no broj trenutnog cvora, parent od roditelja
        self.c_no = c_no
        self.c_id = c_id
        self.f_value = f_value
        self.h_value = h_value
        self.parent_id = parent_id

class Tree:
    def __init__(self):
        self.nodes = {}#cuvaj sve cvorove
    #data, podaci koji se koriste kreiranje novog cvora treenode
    def create_node(self, name, node_id, parent=None, data=None):
        new_node = TreeNode(data.c_no, data.c_id, data.f_value, data.h_value, parent)
        self.nodes[node_id] = new_node
#preuzimanje na osnovu Id
    def get_node(self, node_id):
        return self.nodes.get(node_id)
    
#ostali cvorovi
class FringeNode:
    def __init__(self, c_no, f_value):
        self.f_value = f_value
        self.c_no = c_no

#Prim alg , na init postavljamo matricu susedstva na pocetku  0, a vertices je broj cvorova
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
    
    #nalazi minimalni cvor koji ce se doda u mst-minimalno obuhvatno stablo i dodaje u primov
    def minKey(self, key, mstSet):
        min = sys.maxsize
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def primMST(self, g, d_temp, t):
    #ova lista sadrzi minimalne trenutne vrednosti kljuceva za svaki cvor
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0#postavljamo da je prvi cvor ovradjen
        mstSet = [False] * self.V # ovo ukljucuje da su svi cvorovi u minimalno obuhvatno stablo, a mst nam je lista koju inicijalizujemo, a self.V nam je duzina
        parent[0] = -1#roditelja smo postavili na -1 jer prvi cvor nema roditelja

        for c in range(self.V):
            u = self.minKey(key, mstSet) # pozivamo funkciju minkey da pronadje sa najmanjim kljucem koji u mst, a U ce sadrzati index tog cvora 
            mstSet[u] = True
            for v in range(self.V):#ulazimo u petlju koja ce izvrsiti svaki cvor v u grafu
                #proveravamo da li postoji grana izmedju u i v jer smo ukljucili v u mst preko u 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        #racunamo ukupnu tezinu mst tako sto sumiramo sve grane izmedju cvorova i njigove roditelje
        #pocevsi od indexa 1 jer pocetni cvor nema roditelja
        sum_weight = sum(self.graph[i][parent[i]] for i in range(1, self.V))
        return sum_weight


def heuristic(tree, p_id, t, V, graph):
    visited = set()    
    #dodajemo pocetni cvor u 0 u set posetio 
    visited.add(0)
    visited.add(t)#ovde dodajemo ciljni 
    #proveravamo da li postoji roditelj p_id, ako postoji ulazimo u petlju za racunanje svih cvorova 
    #koje smo posetili od putem od ciljnog ka pocetnom
    if p_id != -1:
        #ovde dobijamo cvor sa id iz stabla pretrage
        tnode = tree.get_node(str(p_id))
        while tnode.c_id != 1:#ulazimo u petlju koja ce se izvrsiti sve dok ne stignemo do pocetnog cvora, da nam je c_id = 1
            visited.add(tnode.c_no)
            parent_id = tnode.parent_id
            tnode = tree.get_node(str(parent_id))

    #racunamo broj cvorova koje smo posetili i cuvamo u promenljivu L
    l = len(visited)
    num = V - l #ovde racunamo broj preostalih koje treba posetiti
    if num != 0:  #ovde proveravamo da li postoji cvor koji treba posetiti     
        g = Graph(num)
        d_temp = {i: key for key, i in enumerate(set(range(V)) - visited)} #ovde mapiramo indexe cvorova koje smo posetili 


        for i in range(V):
            for j in range(V):
                if i not in visited and j not in visited:
                    g.graph[d_temp[i]][d_temp[j]] = graph[i][j]

        mst_weight = g.primMST(graph, d_temp, t)
        #pozivamo primov algoritam koko vi izracunali minimalnu vrednost za preostale cvorove
        #ovo se radi kako bi dobili heuristicku vrednost

        return mst_weight
    else:
        return graph[t][0]

	 
def checkPath(tree, toExpand, V):	 #proveravamo da li je putanja validna od pocetnog cvora
    tnode = toExpand    #postavljamo trenutn cvor  na cvor koji zelimo da proverimo toExpand
    list1 = list()    #list1 nam sluzi za cuvanje indexa cvorova koji  cine putanju od trenutnog do pocetnog

    if tnode.c_id == 1:
        return 0
    else:    #ako trenutni cvor nije pocetni ulazimo u ovu obradu
        depth = 0
        #na S inicijalizujemo indexe cvorova koje smo posetili tokom pracenja putanje
        s = set()
        while tnode.c_id != 1:        #izvrsava se sve dok ne dodjemo do pocetnog cvora
            s.add(tnode.c_no)
            list1.append(tnode.c_no)
            parent_id = tnode.parent_id            #dobijamo identiikator roditelja kojeg smo posetili 
            tnode = tree.get_node(str(parent_id))
            depth += 1
        list1.append(0)   #dodajemo index pocetnog cvora u listu kako bismo oznacili kraj putanje

        #ako je dubina jednaka broju cvorova V, da li smo posetili sve cvorove S
        #i da li se zavrsava sa indexom 0. Ako su uslovi ispunjeni, validna je putanja
        if depth == V and len(s) == V and list1[0] == 0:
            return 1
        else:
            return 0
		
    #rekonstrukcija celokupen putanje od trenutnog do pocetnog	
    #current_node je trenutni cvor
    #vracamo putanju kao listu indexa cvorova, ali obrnuto poredjane jer smo putanju pravili unazad
    #okretanje putanje ce vratiti ispravan redosled
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
    #promenljiva da li je cilj algoritma postignut
    goalState = 0
    #trenutni cvor koji razmatramo, i u svakoj iteraciji azuriramo ga
    toExpand = TreeNode(0, 0, 0, 0, 0)
    key = 1
    #izracunavamo heuristicku vrednost za pocetno stanje, i argumenti za prosledjivanje koji su u zagradi 
    heu = heuristic(tree, -1, 0, V, graph)
    #-1 postavljamo da nema roditelja 
    tree.create_node("1", "1", parent=-1, data=TreeNode(0, 1, heu, heu, -1))
    #cvorovi koji su kandidati za obradu, od pocetnog cvora
    fringe_list = {key: FringeNode(0, heu)}
    key += 1
    full_path = []#full_path sadrzi konacnu putanju nakon sto se algoritam zavrsi

    #izvrsava sve dok ne dostignemo ciljno stanje
    while not goalState:
        #inicijalizujemo minf na max stanje
        minf = sys.maxsize
        for i in fringe_list.keys():
            if fringe_list[i].f_value < minf:
                #odabiramo cvor sa najmanjom vrednoscu funcije F
                current_fringe_node = fringe_list[i]
                current_tree_node_id = i
                minf = current_fringe_node.f_value

        #dobijamo heiristicku vrednost cvora current, i dodeljujemo ga u h
        current_tree_node = tree.get_node(str(current_tree_node_id))
        h = current_tree_node.h_value
        val = current_fringe_node.f_value - h
        #proveravamo da li je putanja od trenutnog cvora do pocetnog validna
        path = checkPath(tree, current_tree_node, V)
        
        #proveravamo da li je trenutni cvor pocetni cvor,i da li je putanja validna, ako jeste, dostigli smo cilj
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
