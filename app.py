import streamlit as st
import heapq
import pandas as pd
import pydeck as pdk

# --- CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA LINHA DO STREAMLIT) ---
st.set_page_config(page_title="Google Maps Protótipo IA", page_icon="🗺️", layout="wide")

# --- MENU LATERAL DE NAVEGAÇÃO (SISTEMA DE PÁGINAS) ---
st.sidebar.title("Navigation Menu")
pagina = st.sidebar.radio("Selecione a tela:", ["🗺️ Protótipo GPS", "📄 Sobre o Projeto"])

# --- CLASSES BASE (NÓ E PROBLEMA) ---
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

class ProblemaAnhembiGeografico:
    def __init__(self, inicial, objetivo):
        self.estado_inicial = inicial
        self.estado_objetivo = objetivo
        
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
        
        self.coordenadas_reais = {
            'Anhembi_Paulista': (-23.5568, -46.6625), 'Shopping_C3': (-23.5560, -46.6618), 'MASP': (-23.5615, -46.6559),
            'Theatro_Municipal': (-23.5453, -46.6388), 'Estacao_da_Luz': (-23.5364, -46.6340), 'Liberdade': (-23.5552, -46.6338),
            'Praca_da_Se': (-23.5505, -46.6333), 'Catedral_da_Se': (-23.5512, -46.6343), 'Patio_do_Colegio': (-23.5488, -46.6329),
            'Mercadao': (-23.5417, -46.6293), 'Parque_Dom_Pedro': (-23.5447, -46.6284), 'Estacao_Bras': (-23.5475, -46.6163),
            'Museu_Imigracao': (-23.5492, -46.6125), 'Armarinhos_Fernando': (-23.5539, -46.6069), 'Anhembi_Mooca': (-23.5550, -46.6105)
        }

    def acoes(self, estado):
        return list(self.conexoes.get(estado, {}).keys())

    def custo_passo(self, estado, acao):
        return self.conexoes[estado][acao]

    def teste_objetivo(self, estado):
        return estado == self.estado_objetivo

    def calcular_heuristica(self, estado):
        lat1, lon1 = self.coordenadas_reais[estado]
        lat2, lon2 = self.coordenadas_reais[self.estado_objetivo]
        return (((lat1 - lat2) ** 2) + ((lon1 - lon2) ** 2)) ** 0.5 * 111.0

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

# --- CONTROLE DE FLUXO DE PÁGINAS ---

if pagina == "🗺️ Protótipo GPS":
    # --- CONTEÚDO DA PRIMEIRA TELA: O MAPA ---
    st.title("🗺️ Protótipo Google Maps Avançado - Roteamento $A^*$")
    st.write("Visualização de percurso sequencial e alfinetes customizados em tempo real.")

    locais = [
        'Anhembi_Paulista', 'Shopping_C3', 'MASP', 'Theatro_Municipal', 
        'Estacao_da_Luz', 'Liberdade', 'Praca_da_Se', 'Catedral_da_Se', 
        'Patio_do_Colegio', 'Mercadao', 'Parque_Dom_Pedro', 'Estacao_Bras', 
        'Museu_Imigracao', 'Armarinhos_Fernando', 'Anhembi_Mooca'
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuração do Trajeto")
        origem = st.selectbox("📍 Ponto de Origem:", locais, index=0)
        destino = st.selectbox("🏁 Ponto de Destino:", locais, index=14)
        calcular = st.button("🚀 Calcular Rota Otimizada", use_container_width=True)

    with col2:
        if calcular:
            if origem == destino:
                st.warning("A origem e o destino são iguais, escolha locais diferentes!")
            else:
                waze_sp = ProblemaAnhembiGeografico(origem, destino)
                rota, esforco = busca_a_estrela(waze_sp)
                
                if rota:
                    custo_total = rota[-1][1]
                    st.success(f"**Sucesso!** Rota calculada pelo A*: **{custo_total:.2f} km**")
                    
                    dados_pontos = []
                    coordenadas_linha = []
                    
                    for i, (ponto, _) in enumerate(rota):
                        lat, lon = waze_sp.coordenadas_reais[ponto]
                        coordenadas_linha.append([lon, lat])
                        
                        if i == 0:
                            cor_rgb = [46, 204, 113]      # Verde
                        elif i == len(rota) - 1:
                            cor_rgb = [231, 76, 60]       # Vermelho
                        else:
                            cor_rgb = [52, 152, 219]      # Azul
                            
                        dados_pontos.append({
                            "name": ponto, "latitude": lat, "longitude": lon, "color": cor_rgb
                        })
                    
                    df_pontos = pd.DataFrame(dados_pontos)
                    df_linha = pd.DataFrame([{"path": coordenadas_linha}])
                    
                    camada_linha = pdk.Layer(
                        "PathLayer", df_linha, get_path="path", width_min_pixels=4,
                        get_color=[44, 62, 80], pickable=True
                    )
                    
                    camada_pontos = pdk.Layer(
                        "ScatterplotLayer", df_pontos, get_position="[longitude, latitude]",
                        get_radius=120, get_fill_color="color", pickable=True
                    )
                    
                    lat_centro, lon_centro = waze_sp.coordenadas_reais[origem]
                    estado_visao = pdk.ViewState(
                        latitude=lat_centro, longitude=lon_centro, zoom=13, pitch=30
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        layers=[camada_linha, camada_pontos],
                        initial_view_state=estado_visao,
                        tooltip={"text": "Local: {name}"}
                    ))
                    
                    st.subheader("📋 Sequência de Navegação:")
                    st.write(" ➡️ ".join([f"`{p[0]}`" for p in rota]))
                else:
                    st.error("Infelizmente o algoritmo não encontrou conexão direta para o trajeto desejado.")

elif pagina == "📄 Sobre o Projeto":
    # --- CONTEÚDO DA SEGUNDA TELA: A EXPLICAÇÃO DO PROJETO ---
    st.title("📄 Documentação do Projeto: GPS com Algoritmo $A^*$")
    st.write("---")
    
    st.subheader("🤖 Contexto do Desafio Acadêmico")
    st.markdown("""
    Este projeto foi desenvolvido como resposta ao desafio proposto em ambiente de laboratório para mapear e encontrar a rota ótima entre as unidades da **Universidade Anhembi Morumbi (Campus Paulista para o Campus Mooca)**.
    Em vez de utilizarmos uma abordagem puramente teórica (como o tradicional mapa da Romênia contido no livro de *Russell & Norvig*), trouxemos o problema para a malha urbana real da cidade de São Paulo.
    """)
    
    st.subheader("🧠 O Algoritmo Usado: Busca $A^*$ (A-Estrela)")
    st.markdown("""
    O sistema adota a **Busca Informada $A^*$**, que avalia os nós combinando o custo real acumulado do trajeto com uma estimativa heurística de distância até o objetivo final.
    A função de avaliação é regida pela famosa fórmula matemática:
    """)
    
    st.latex(r"f(n) = g(n) + h(n)")
    
    st.markdown("""
    * **$g(n)$ (Custo Real):** Representa a distância real em quilômetros das vias asfalto que conectam um marco turístico ou avenida até o próximo nó.
    * **$h(n)$ (Heurística Admissível):** É calculada de forma dinâmica utilizando a distância euclidiana em linha reta (coordenadas geográficas reais de latitude e longitude) entre o nó atual e o destino selecionado pelo usuário. Por nunca superestimar a distância real de malha rodoviária, essa heurística é estritamente **admissível**, garantindo que o algoritmo encontre sempre o caminho ótimo.
    """)
    
    st.subheader("💻 Arquitetura Tecnológica")
    st.markdown("""
    * **Backend:** Programação Orientada a Objetos (POO) em Python, utilizando estruturas nativas de tuplas, conjuntos (`set`) para a lista fechada (explorados) e o módulo de filas de prioridade de baixo nível `heapq` para a expansão eficiente da borda.
    * **Frontend:** Interface Web reativa criada inteiramente através do framework **Streamlit**, permitindo interatividade dinâmica através de coleções e renderização avançada de mapas geográficos com mapas em múltiplas camadas através do motor **Pydeck**.
    """)
    
    st.info("💡 **Dica para a Apresentação:** Use o menu lateral à esquerda para alternar livremente entre a simulação interativa no mapa de São Paulo e esta tela de fundamentação teórica!")
