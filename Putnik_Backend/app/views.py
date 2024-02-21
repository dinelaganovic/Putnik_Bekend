from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from app.Pytnik.agents import izdvoji_putanje, broj_redova_u_stringu, izdvoji_zlatnike, Aki, Jocke, Uki, Micko, zbirPutanjaAki
from app.Pytnik.agents import opisPutanjaMat, zbirJocke, Tree


@csrf_exempt
def handle_map_and_agent(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        map_content = data['mapContent']
        agent_index = data['agentIndex']

        zlatnici = izdvoji_zlatnike(map_content)
        matricaPutanja = izdvoji_putanje(map_content)
        brRedova = broj_redova_u_stringu(map_content)
        opisPutanja = []
        ukupanZbir = ""
        agent = []
        simpleTree = Tree()
        
        if agent_index == "0":
            agent = Aki(zlatnici,matricaPutanja)
            opisPutanja = opisPutanjaMat(matricaPutanja,agent)
            ukupanZbir = zbirPutanjaAki(zlatnici,matricaPutanja)
        elif agent_index == "1":
            agent = Jocke(matricaPutanja)
            ukupanZbir = zbirJocke(zlatnici,matricaPutanja)
            opisPutanja = opisPutanjaMat(matricaPutanja,agent)
        elif agent_index == "2":            
            path, zbir = Uki(matricaPutanja,brRedova) 
            agent = path
            ukupanZbir = zbir
            opisPutanja = opisPutanjaMat(matricaPutanja, agent)
        elif agent_index == "3":
            zbir, path = Micko(matricaPutanja,simpleTree,brRedova)
            ukupanZbir = zbir
            agent = path
            opisPutanja = opisPutanjaMat(matricaPutanja,agent)
        
        response_data = {
            'agentIndex': agent_index,
            "putanje": matricaPutanja,
            "agent": agent,
            "opisPutanja" : opisPutanja,
            "zbir": ukupanZbir
        }
        
        return JsonResponse({'data': response_data})
    return JsonResponse({'error': 'Neispravan zahtev'}, status=400)
