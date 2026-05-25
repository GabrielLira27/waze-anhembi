import streamlit as st
import heapq
import pandas as pd
import numpy as np
import pydeck as pdk
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA LINHA DO STREAMLIT) ---
st.set_page_config(page_title="Portal Avançado de IA - Grupo", page_icon="⚡", layout="wide")

# --- MENU LATERAL DE NAVEGAÇÃO ---
st.sidebar.title("🤖 Painel de Controle de IA")
st.sidebar.markdown("Selecione qual módulo da nossa equipe você deseja auditar:")
pagina = st.sidebar.radio("Navegar para:", ["🗺️ Protótipo GPS (A*)", "⚡ Inteligência Preditiva (Crypto)", "📄 Sobre o Projeto"])

# ==============================================================================
# MÓDULO 1: O SEU PROJETO DE ROTAS (A*)
# ==============================================================================
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

    def acoes(self, estado): return list(self.conexoes.get(estado, {}).keys())
    def custo_passo(self, estado, acao): return self.conexoes[estado][acao]
    def teste_objetivo(self, estado): return estado == self.estado_objetivo
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
        if problema.teste_objetivo(no_atual.estado): return reconstruir_caminho(no_atual), nos_expandidos
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


# ==============================================================================
# EXECUÇÃO DA PÁGINA CORRESPONDENTE
# ==============================================================================

if pagina == "🗺️ Protótipo GPS (A*)":
    st.title("🗺️ Protótipo Google Maps Avançado - Roteamento $A^*$")
    st.write("Mapeamento sequencial de rotas otimizadas por IA geolocalizada em São Paulo.")
    
    locais = list(ProblemaAnhembiGeografico('Anhembi_Paulista', 'Anhembi_Mooca').coordenadas_reais.keys())
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuração do Trajeto")
        origem = st.selectbox("📍 Ponto de Origem:", locais, index=0)
        destino = st.selectbox("🏁 Ponto de Destino:", locais, index=14)
        calcular = st.button("🚀 Calcular Rota Otimizada", use_container_width=True)
    
    with col2:
        if calcular:
            if origem == destino:
                st.warning("A origem e o destino são iguais!")
            else:
                waze_sp = ProblemaAnhembiGeografico(origem, destino)
                rota, esforc = busca_a_estrela(waze_sp)
                if rota:
                    custo_total = rota[-1][1]
                    st.success(f"**Sucesso!** Caminho calculado: **{custo_total:.2f} km**")
                    
                    coordenadas_linha = []
                    dados_pontos = []
                    for i, (ponto, _) in enumerate(rota):
                        lat, lon = waze_sp.coordenadas_reais[ponto]
                        coordenadas_linha.append([lon, lat])
                        cor = [46, 204, 113] if i == 0 else ([231, 76, 60] if i == len(rota)-1 else [52, 152, 219])
                        dados_pontos.append({"name": ponto, "latitude": lat, "longitude": lon, "color": cor})
                    
                    df_pontos = pd.DataFrame(dados_pontos)
                    df_linha = pd.DataFrame([{"path": coordenadas_linha}])
                    
                    st.pydeck_chart(pdk.Deck(
                        layers=[
                            pdk.Layer("PathLayer", df_linha, get_path="path", width_min_pixels=4, get_color=[44, 62, 80]),
                            pdk.Layer("ScatterplotLayer", df_pontos, get_position="[longitude, latitude]", get_radius=120, get_fill_color="color", pickable=True)
                        ],
                        initial_view_state=pdk.ViewState(latitude=waze_sp.coordenadas_reais[origem][0], longitude=waze_sp.coordenadas_reais[origem][1], zoom=12, pitch=30),
                        tooltip={"text": "Local: {name}"}
                    ))
                    st.subheader("📋 Sequência de Navegação:")
                    st.write(" ➡️ ".join([f"`{p[0]}`" for p in rota]))

elif pagina == "⚡ Inteligência Preditiva (Crypto)":
    # ==============================================================================
    # MÓDULO 2: O PROJETO DE CRIPTO DO SEU AMIGO (TREINAMENTO + COMPILADOR REAL-TIME)
    # ==============================================================================
    st.title("⚡ Neural Crypto Enterprise v4.0")
    st.write("Módulo preditivo baseado no algoritmo Random Forest Regressor e Indicadores de Força Relativa (RSI).")
    
    ativos = {'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD'}
    moeda = st.selectbox("🪙 Escolha a Criptomoeda para análise preditiva:", list(ativos.keys()))
    
    with st.spinner(f"📡 Buscando dados históricos em tempo real do {moeda} no Yahoo Finance..."):
        # Baixa os dados em tempo real direto pela nuvem
        df = yf.download(ativos[moeda], period='max', progress=False)
        
        # --- CORREÇÃO DO BUG DO INDEX/DATE ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Força o reset do índice de forma segura e renomeia para 'Date' caso mude de nome
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'Date'})
        elif 'Date' not in df.columns:
            # Caso o índice original estivesse sem nome, ele pode ter virado outra coisa
            df.columns.values[0] = 'Date'
        # -------------------------------------
        
        # Engenharia de Atributos que ele montou
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        delta = df['Close'].diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (ganho / perda)))
        df = df.dropna().reset_index(drop=True)
        
        # Prepara dados para o modelo dele rodar ao vivo
        colunas_ia = ['Close', 'SMA_7', 'SMA_21', 'RSI']
        scaler = MinMaxScaler()
        dados_norm = scaler.fit_transform(df[colunas_ia])
        
        X, y = [], []
        for i in range(14, len(dados_norm) - 5):
            X.append(dados_norm[i-14:i].flatten())
            y.append(df['Close'].iloc[i + 5])
        
        X, y = np.array(X), np.array(y)
        
        # Divisão de Validação Cega
        split = int(len(X) * 0.9)
        X_treino, X_val = X[:split], X[split:]
        y_treino, y_val = y[:split], y[split:]
        
        # Treina a árvore de decisão Random Forest do seu parceiro na hora
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_treino, y_treino)
        
        # Projeções
        previsoes = modelo.predict(X_val)
        datas_val = df['Date'].iloc[split + 14 + 5 : len(df)].values
        reais = y_val
        
    # Exibição das Métricas Visuais dele
    col1, col2, col3 = st.columns(3)
    preco_atual = float(df['Close'].iloc[-1])
    col1.metric(label=f"Preço Atual {moeda}", value=f"${preco_atual:,.2f}")
    col2.metric(label="Algoritmo de Aprendizado", value="Random Forest")
    col3.metric(label="Indicadores Técnicos", value="RSI + SMA (7,21)")
    
    # Renderiza o gráfico Plotly interativo brabo dele
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datas_val, y=reais, mode='lines', name='Preço Real', line=dict(color='#00e5ff', width=3)))
    fig.add_trace(go.Scatter(x=datas_val, y=previsoes, mode='lines', name='Projeção IA', line=dict(color='#ff9900', width=2.5, dash='dash')))
    fig.update_layout(title=f"Auditoria Gráfica Neon: Projeções do Modelo para o {moeda}", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif pagina == "📄 Sobre o Projeto":
    st.title("📄 Documentação do Projeto Unificado do Grupo")
    st.write("---")
    st.subheader("👨‍💻 Soluções Integradas da Equipe")
    st.markdown("""
    O presente portal foi unificado via Git para expor duas vertentes complementares de Inteligência Artificial aplicadas a problemas reais:
    
    1. **Planejamento de Trajetos Urbanos ($A^*$):** Algoritmo de busca informada geolocalizado que calcula a menor rota física viável de carro entre os campi Paulista e Mooca da Anhembi Morumbi, mitigando o processamento computacional através de uma heurística euclidiana estritamente admissível.
    2. **Análise Preditiva de Séries Temporais (Random Forest):** Uma IA baseada em aprendizado supervisionado (Machine Learning) estruturada com florestas de decisão para prever as oscilações de fechamento no mercado hipervolátil de criptoativos de grande capitalização.
    """)
    st.success("🚀 Trabalho integrado com sucesso! O menu lateral gerencia os estados de visualização de forma limpa na nuvem.")
