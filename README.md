# 🌍 Implementação Otimizada de PCA para Análise Climática (ENSO / El Niño)

Este projeto de Física Computacional demonstra a implementação matemática do algoritmo de **Análise de Componentes Principais (PCA)** "do zero" (from scratch) para processamento de dados multidimensionais, focado na identificação de padrões climáticos globais (El Niño-Southern Oscillation).

## 🎯 O Problema e Diferencial Técnico
Muitos cientistas de dados utilizam o PCA como uma "caixa preta" via Scikit-Learn. Neste projeto, o objetivo foi ir a fundo na **Álgebra Linear** por trás do algoritmo, construindo a fatoração e a projeção ortogonal a partir da matriz de covariância. 

Para lidar com o alto custo computacional de matrizes climáticas gigantes (dados espaço-temporais), a solução integra compilação JIT (Just-In-Time) para acelerar cálculos matriciais em baixo nível, provando forte capacidade em Engenharia de Software e Matemática Aplicada.

## 🚀 Destaques da Arquitetura
1. **PCA Matemático (NumPy):** Cálculo exato de autovalores e autovetores (`np.linalg.eigh`) sobre a matriz de covariância, com ordenação manual para capturar a variância máxima (EOF1).
2. **Otimização de Performance (Numba):** Utilização do decorador `@njit` para compilar as operações de multiplicação de matrizes em código de máquina, reduzindo drasticamente o tempo de cálculo dos *scores*.
3. **Engenharia de Dados Multidimensionais:** Uso do `xarray` para empilhar e desempilhar coordenadas geoespaciais (latitude e longitude) e temporais.
4. **Heurística de Correção de Sinal:** Algoritmo de verificação de coordenadas específicas (Pacífico Leste / Região Niño 3.4) para garantir a consistência física do padrão (sinal positivo) antes da plotagem.

## 🛠️ Tecnologias Utilizadas
* **NumPy:** Álgebra linear e operações matriciais vetoriais.
* **Numba (JIT):** Aceleração de funções críticas de cálculo matricial.
* **Xarray:** Manipulação eficiente de datasets multidimensionais (formato NetCDF).
* **Cartopy & Matplotlib:** Projeções geoespaciais e visualização avançada de mapas e séries temporais.

## ⚙️ Como Executar
1. Clone o repositório.
2. Instale as dependências:
  ```bash
   pip install numpy xarray matplotlib cartopy numba
