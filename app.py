import streamlit as st
import heapq

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
class ProblemaAnhembiDinamico:
    def __init__(self, inicial, objetivo):
        self.estado_inicial = inicial
        self.estado_objetivo = objetivo
        
        # O grafo completo com os seus pontos turísticos de SP
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
        
        # Como o destino agora muda na interface, criamos uma Heurística Geral aproximada
        # Representa a distância estimada em linha reta entre os pontos
        self.coordenadas_estimadas = {
            'Anhembi_Paulista': (0, 0), 'Shopping_C3': (0, 0.2), 'MASP': (-0.3, 0),
            'Theatro_Municipal': (2, 2), 'Estacao_da_Luz': (3, 1.5), 'Liberdade': (1.8, 1),
            'Praca_da_Se': (2.2, 1.8), 'Catedral_da_Se': (2.3, 1.8), 'Patio_do_Colegio': (2.4, 2),
            'Mercadao': (2.8, 2.3), 'Parque_Dom_Pedro': (2.7, 1.5), 'Estacao_Bras': (3.5, 2.5),
            'Museu_Imigracao': (4, 3), 'Armarinhos_Fernando': (4.5, 3.2), 'Anhembi_Mooca': (5, 3.5)
        }

    def acoes(self, estado):
        return list(self.conexoes.get(estado, {}).keys())

    def custo_passo(self, estado, acao):
        return self.conexoes[estado][acao]

    def teste_objetivo(self, estado):
        return estado == self.estado_objetivo

    def calcular_heuristica(self, estado):
        # Calcula a distância Euclidiana simples entre o ponto atual e o destino final escolhido
        p1 = self.coordenadas_estimadas[estado]
        p2 = self.coordenadas_estimadas[self.estado_objetivo]
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

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
                heapq.heappush(borda, child=filho)
    return None, nos_expandidos

def reconstruir_caminho(no):
    caminho = []
    while no:
        caminho.append((no.estado, no.custo_caminho))
        no = no.pai
    return caminho[::-1]

# --- 4. INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(page_title="Waze Real-Time Inteligente", page_icon="🗺️")

st.title("🗺️ Protótipo Google Maps - Roteamento IA")
st.write("Selecione os pontos do campus e locais de SP para rodar a Busca $A^*$ em tempo real.")

# Lista de locais disponíveis no seu sistema
locais = [
    'Anhembi_Paulista', 'Shopping_C3', 'MASP', 'Theatro_Municipal', 
    'Estacao_da_Luz', 'Liberdade', 'Praca_da_Se', 'Catedral_da_Se', 
    'Patio_do_Colegio', 'Mercadao', 'Parque_Dom_Pedro', 'Estacao_Bras', 
    'Museu_Imigracao', 'Armarinhos_Fernando', 'Anhembi_Mooca'
]

# Caixas de Seleção Estilizadas
col1, col2 = st.columns(2)
with col1:
    origem = st.selectbox("📍 Escolha o ponto de partida:", locais, index=0)
with col2:
    destino = st.selectbox("🏁 Escolha o destino final:", locais, index=14)

if st.button("🚀 Calcular Melhor Rota de Carro", use_container_width=True):
    if origem == destino:
        st.warning("Mano, você já está no destino! Escolha pontos diferentes. 😅")
    else:
        # Inicializa e roda o problema com as escolhas da tela
        waze_sp = ProblemaAnhembiDinamico(origem, destino)
        rota, esforco = busca_a_estrela(waze_sp)
        
        if rota:
            st.success("🎉 Rota excelente encontrada pelo algoritmo!")
            
            # Caixa com o resumo do trajeto
            custo_total = rota[-1][1]
            st.metric(label="Distância Total da Viagem", value=f"{custo_total:.2f} km")
            st.metric(label="Esforço de Processamento (Nós Expandidos)", value=f"{esforco} locais analisados")
            
            # Mostra o passo a passo bonitinho
            st.subheader("📋 Instruções do Trajeto:")
            for i, (ponto, km) in enumerate(rota):
                if i == 0:
                    st.write(f"➡️ **Parta de:** `{ponto}`")
                elif i == len(rota) - 1:
                    st.write(f"🏁 **Chegue em:** `{ponto}` (Distância acumulada: *{km:.2f} km*)")
                else:
                    st.write(f"📍 Siga para: `{ponto}` (+{km:.2f} km)")
        else:
            st.error("Desculpe, meu chapa. Não há caminhos disponíveis entre esses pontos no momento.")
