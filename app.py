import streamlit as st
import heapq
import pandas as pd

# --- 1. CLASSE NÓ ---
class No:
    def __init__(self, estado, pai=None, acao=None, custo_caminho=0, heuristica=0):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.custo_caminho = custo_caminho
        self.heuristica = heuristica
        self.f = custo_caminho + heuristica

    def __lt__(self, outro):
        return self.f < outro.f

# --- 2. CLASSE PROBLEMA DINÂMICO ---
class ProblemaAnhembiGeografico:
    def __init__(self, inicial, objetivo):
        self.estado_inicial = inicial
        self.estado_objetivo = objetivo
        
        # Grafo de conexões (Distâncias aproximadas de trânsito em km)
        self.conexoes = {
            'Anhembi_Paulista': {'Shopping_C3': 0.2, 'MASP': 0.5, 'Theatro_Municipal': 4.0},
            'Shopping_C3': {'Anhembi_Paulista': 0.2, 'Theatro_Municipal': 3.8},
            'MASP': {'Anhembi_Paulista': 0.5, 'Theatro_Municipal': 3.5, 'Liberdade': 3.2},
            'Theatro_Municipal': {'Shopping_C3': 3.8, 'MASP': 3.5, 'Estacao_da_Luz': 1.8, 'Praca_da_Se': 1.0},
            'Estacao_da_Luz': {'Theatro_Municipal': 1.8, 'Mercadao': 1.1},
            'Liberdade': {'MASP': 3.2, 'Praca_da_Se': 1.1, 'Parque_Dom_Pedro': 1.8},
            'Praca_da_Se': {'Theatro_Municipal': 1.0, 'Catedral_da_Se': 0.1, 'Liberdade': 1.1, 'Patio_do_Colegio': 0.4},
            'Catedral_da_Se': {'Praca_da_Se': 0.1, 'Parque_Dom_Pedro': 1.2},
            'Patio_do_Colegio': {'Praca_da_Se': 0.4, 'Mercadao': 1.0, 'Parque_Dom_Pedro': 0.8},
            'Mercadao': {'Estacao_da_Luz': 1.1, 'Patio_do_Colegio': 1.0, 'Parque_Dom_Pedro': 0.9, 'Museu_Imigracao': 2.8},
            'Parque_Dom_Pedro': {'Catedral_da_Se': 1.2, 'Patio_do_Colegio': 0.8, 'Mercadao': 0.9, 'Estacao_Bras': 1.5, 'Museu_Imigracao': 2.2},
            'Estacao_Bras': {'Parque_Dom_Pedro': 1.5, 'Museu_Imigracao': 1.3, 'Anhembi_Mooca': 2.1},
            'Museu_Imigracao': {'Mercadao': 2.8, 'Parque_Dom_Pedro': 2.2, 'Estacao_Bras': 1.3, 'Theatro_Municipal': 4.5, 'Armarinhos_Fernando': 1.5},
            'Armarinhos_Fernando': {'Museu_Imigracao': 1.5, 'Anhembi_Mooca': 0.8},
            'Anhembi_Mooca': {'Armarinhos_Fernando': 0.8, 'Estacao_Bras': 2.1}
        }
        
        # COORDENADAS REAIS DE SÃO PAULO (Latitude, Longitude)
        self.coordenadas_reais = {
            'Anhembi_Paulista': (-23.5568, -46.6625),
            'Shopping_C3': (-23.5560, -46.6618),
            'MASP': (-23.5615, -46.6559),
            'Theatro_Municipal': (-23.5453, -46.6388),
            'Estacao_da_Luz': (-23.5364, -46.6340),
            'Liberdade': (-23.5552, -46.6338),
            'Praca_da_Se': (-23.5505, -46.6333),
            'Catedral_da_Se': (-23.5512, -46.6343),
            'Patio_do_Colegio': (-23.5488, -46.6329),
            'Mercadao': (-23.5417, -46.6293),
            'Parque_Dom_Pedro': (-23.5447, -46.6284),
            'Estacao_Bras': (-23.5475, -46.6163),
            'Museu_Imigracao': (-23.5492, -46.6125),
            'Armarinhos_Fernando': (-23.5539, -46.6069),
            'Anhembi_Mooca': (-23.5550, -46.6105)
        }

    def acoes(self, estado):
        return list(self.conexoes.get(estado, {}).keys())

    def custo_passo(self, estado, acao):
        return self.conexoes[estado][acao]

    def teste_objetivo(self, estado):
        return estado == self.estado_objetivo

    def calcular_heuristica(self, estado):
        # Haversine simplificada ou distância euclidiana direta sobre graus geográficos
        # Funciona perfeitamente como estimativa em linha reta para distâncias curtas urbanas
        lat1, lon1 = self.coordenadas_reais[estado]
        lat2, lon2 = self.coordenadas_reais[self.estado_objetivo]
        # Multiplicamos por 111 para converter aproximadamente graus em km
        return (((lat1 - lat2) ** 2) + ((lon1 - lon2) ** 2)) ** 0.5 * 111.0

# --- 3. MOTOR A* ---
def busca_a_estrela(problema):
    borda = []
    h_inicial = problema.calcular_heuristica(problema.estado_inicial)
    no_inicial = No(problema.estado_inicial, heuristica=h_inicial)
    heapq.heappush(borda, no_inicial)
    
    explorado = set()
    nos_expandidos = 0
    
    while borda:
        no_atual = heapq.heappop(borda)
        nos_expandidos += 1
        
        if problema.teste_objetivo(no_atual.estado):
            return reconstruir_caminho(no_atual), nos_expandidos
            
        explorado.add(no_atual.estado)
        
        for acao in problema.acoes(no_atual.estado):
            vizinho = acao
            custo_g = no_atual.custo_caminho + problema.custo_passo(no_atual.estado, acao)
            
            if vizinho not in explorado:
                h_filho = problema.calcular_heuristica(vizinho)
                filho = No(vizinho, no_atual, vizinho, custo_g, h_filho)
                heapq.heappush(borda, filho)
    return None, nos_expandidos

def reconstruir_caminho(no):
    caminho = []
    while no:
        caminho.append((no.estado, no.custo_caminho))
        no = no.pai
    return caminho[::-1]

# --- 4. INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(page_title="Google Maps Protótipo IA", page_icon="🗺️", layout="wide")

st.title("🗺️ Protótipo Google Maps - Sistema de Roteamento $A^*$")
st.write("Filtre o trajeto do veículo utilizando dados geográficos reais das vias de São Paulo.")

locais = [
    'Anhembi_Paulista', 'Shopping_C3', 'MASP', 'Theatro_Municipal', 
    'Estacao_da_Luz', 'Liberdade', 'Praca_da_Se', 'Catedral_da_Se', 
    'Patio_do_Colegio', 'Mercadao', 'Parque_Dom_Pedro', 'Estacao_Bras', 
    'Museu_Imigracao', 'Armarinhos_Fernando', 'Anhembi_Mooca'
]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuração da Viagem")
    origem = st.selectbox("📍 Ponto de Partida:", locais, index=0)
    destino = st.selectbox("🏁 Destino Final:", locais, index=14)
    calcular = st.button("🚀 Gerar Mapa e Percurso", use_container_width=True)

with col2:
    if calcular:
        if origem == destino:
            st.warning("Você já está no local escolhido! Altere a origem ou o destino.")
        else:
            waze_sp = ProblemaAnhembiGeografico(origem, destino)
            rota, esforco = busca_a_estrela(waze_sp)
            
            if rota:
                custo_total = rota[-1][1]
                st.success(f"**Rota Otimizada Calculada!** Distância total: **{custo_total:.2f} km**")
                
                # --- CONSTRUÇÃO DO MAPA VISUAL ---
                dados_mapa = []
                for ponto, _ in rota:
                    lat, lon = waze_sp.coordenadas_reais[ponto]
                    dados_mapa.append({"name": ponto, "latitude": lat, "longitude": lon})
                
                df_rota = pd.DataFrame(dados_mapa)
                
                # Renderiza o mapa interativo na tela com os pontos do trajeto
                st.map(df_rota, zoom=13, use_container_width=True)
                
                # Histórico textual logo abaixo
                st.subheader("📋 Resumo do Itinerário")
                caminho_formatado = " ➡️ ".join([f"`{p[0]}`" for p in rota])
                st.write(caminho_formatado)
            else:
                st.error("Nenhum trajeto foi encontrado conectando as duas localidades no grafo atual.")
