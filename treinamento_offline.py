import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib 

# Lista das moedas que a nossa startup vai operar
ativos = {
    'BTC': 'BTC-USD', 
    'ETH': 'ETH-USD', 
    'SOL': 'SOL-USD'
}

janela = 14
horizonte = 5

print("⚙️ INICIANDO PIPELINE DE TREINAMENTO MULTI-MOEDAS...")

# O loop vai treinar uma IA separada para cada moeda!
for simbolo, ticker in ativos.items():
    print(f"\n📡 [1/4] Baixando histórico de 3 anos do {simbolo}...")
    df = yf.download(ticker, start='2023-01-01', end='2025-06-30', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    print(f"🧮 [2/4] Calculando RSI e Médias Móveis do {simbolo}...")
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    delta = df['Close'].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna().reset_index(drop=True)
    
    colunas_ia = ['Close', 'SMA_7', 'SMA_21', 'RSI']
    scaler_x = MinMaxScaler()
    dados_x_norm = scaler_x.fit_transform(df[colunas_ia])
    
    scaler_y = MinMaxScaler()
    dados_y_norm = scaler_y.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(janela, len(dados_x_norm) - horizonte):
        X.append(dados_x_norm[i-janela:i].flatten()) 
        y.append(dados_y_norm[i + horizonte, 0])
        
    X, y = np.array(X), np.array(y)
    
    print(f"🧠 [3/4] Treinando a IA Especialista em {simbolo}...")
    modelo = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    
    print(f"💾 [4/4] Salvando o cérebro do {simbolo} em disco...")
    # O nome do arquivo salvo muda de acordo com a moeda!
    joblib.dump(modelo, f'modelo_{simbolo}.pkl')
    joblib.dump(scaler_x, f'scaler_x_{simbolo}.pkl')
    joblib.dump(scaler_y, f'scaler_y_{simbolo}.pkl')

print("\n✅ SUCESSO! Todos os 3 modelos especialistas (BTC, ETH, SOL) foram gerados e salvos na sua pasta.")
