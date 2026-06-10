import h5py
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
caminho_arquivo = 'D:\\Downloads HD\\introdução a inteligencia computacional\\archive\\hif_vegetation_dataset.h5'

# ==========================================
# 2. FUNÇÃO MATEMÁTICA BASE
# ==========================================
def calcular_metricas(sinal, Fs):
    """Calcula RMS, Fator de Crista e Amplitudes da FFT."""
    N = len(sinal)
    
    # --- TEMPO ---
    rms = np.sqrt(np.mean(sinal**2))
    pico_maximo = np.max(np.abs(sinal))
    fator_crista = pico_maximo / rms if rms > 0 else 0
    
    # --- FREQUÊNCIA ---
    fft_valores = np.fft.fft(sinal)
    frequencias = np.fft.fftfreq(N, 1/Fs)
    
    metade = N // 2
    freqs_positivas = frequencias[:metade]
    magnitudes = np.abs(fft_valores)[:metade] * (2.0 / N)
    
    def pegar_amplitude(freq_alvo):
        indice = np.argmin(np.abs(freqs_positivas - freq_alvo))
        return magnitudes[indice]
    
    return {
        'Valor_RMS': rms,
        'Fator_Crista': fator_crista,
        'Amp_50Hz': pegar_amplitude(50),
        'Amp_150Hz': pegar_amplitude(150),
        'Amp_250Hz': pegar_amplitude(250)
    }

# ==========================================
# 3. EXTRAÇÃO EM MASSA
# ==========================================
dados_extraidos = []

print("Abrindo o arquivo HDF5 e iniciando a extração...")

with h5py.File(caminho_arquivo, 'r') as f:
    
    # Vamos varrer a raiz: 'test' (FAI) recebe classe 1, 'cal' (Normal) recebe classe 0
    grupos_principais = [('test', 1), ('cal', 0)]
    
    for nome_grupo, rotulo_classe in grupos_principais:
        grupo_atual = f[nome_grupo]
        
        # .keys() nos dá a lista de pastas lá dentro ('015', '036', '1', etc.)
        for teste_id in grupo_atual.keys():
            pasta_teste = grupo_atual[teste_id]
            
            # Criamos a linha base da nossa tabela
            linha = {
                'Nome_Arquivo': teste_id,
                'Classe': rotulo_classe
            }
            
            # Roteiro do que buscar dentro desta pasta (Sinal, Prefixo na Tabela, Fs)
            sinais_para_buscar = [
                ('current_lf', 'Curr_LF_', 100000),
                ('current_hf', 'Curr_HF_', 2000000),
                ('voltage_lf', 'Volt_LF_', 100000),
                ('voltage_hf', 'Volt_HF_', 2000000)
            ]
            
            for nome_sinal, prefixo, fs in sinais_para_buscar:
                # Segurança: verifica se o sinal existe dentro da pasta do teste
                if nome_sinal in pasta_teste:
                    # Carrega o dado direto do HD convertendo para 1D (linha 0, todas as colunas)
                    sinal_1d = pasta_teste[nome_sinal][0, :]
                    metricas = calcular_metricas(sinal_1d, fs)
                    
                    # Adiciona as métricas com o prefixo para não misturar tensão e corrente
                    for chave, valor in metricas.items():
                        linha[f"{prefixo}{chave}"] = valor
                else:
                    # Se faltar algum sensor naquele teste, preenche com nulo (NaN)
                    for chave in ['Valor_RMS', 'Fator_Crista', 'Amp_50Hz', 'Amp_150Hz', 'Amp_250Hz']:
                        linha[f"{prefixo}{chave}"] = np.nan
            
            # Adiciona a linha finalizada à nossa lista geral
            dados_extraidos.append(linha)

# ==========================================
# 4. CONSOLIDAÇÃO E SALVAMENTO
# ==========================================
df_final = pd.DataFrame(dados_extraidos)

# Removemos qualquer teste que tenha dados corrompidos/faltando em algum dos canais
df_final = df_final.dropna()

df_final.to_csv('base_treinamento_h5.csv', index=False)

print(f"\nExtração Concluída com Sucesso!")
print(f"Total de testes processados perfeitos: {len(df_final)}")
print("\nColunas geradas no seu CSV:")
for col in df_final.columns:
    print(f" - {col}")
