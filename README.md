# Esteira de Machine Learning — Predição de Risco de Consumo de Cannabis

Projeto acadêmico de **Aprendizado de Máquina (Classificação Binária)** que implementa uma esteira completa — da carga e exploração dos dados até o treinamento, avaliação e implantação do modelo em disco — utilizando a base **Drug Consumption (Quantified)** da UCI (ID `373`).

---

## O que é o projeto

O objetivo é prever se uma pessoa apresenta perfil de **usuária** ou **não usuária de cannabis** (consumo na última década ou mais recente) a partir exclusivamente de:

- **Variáveis demográficas:** idade (`age`), gênero (`gender`), escolaridade (`education`), país (`country`) e etnia (`ethnicity`).
- **Traços de personalidade:** modelo dos Cinco Fatores NEO-FFI-R (`nscore`, `escore`, `oscore`, `ascore`, `cscore`), impulsividade (`impulsive`) e busca de sensação (`ss`), além da variável combinada `ss_impulsivo = ss + impulsive`.

As demais 18 substâncias presentes na base original são removidas das preditoras para evitar vazamento de dados (*data leakage*).

---

## Estrutura do repositório

```text
Machine-Learning-Drug-Consumption/
├── esteira_classificacao_drug_consumption.ipynb  # Notebook completo com toda a esteira
├── drug+consumption+quantified.zip               # Fallback offline da base UCI (id=373)
├── requirements.txt                              # Dependências Python do projeto
├── .gitignore                                    # Arquivos ignorados pelo Git (.venv, .joblib)
└── README.md                                     # Documentação e instruções de execução
```

> Ao executar o notebook, a pasta `artefatos/` é criada automaticamente com o modelo treinado (`modelo_cannabis.joblib`) e o contrato de entrada (`schema_entrada.json`).

---

## Etapas da esteira

1. **Carga da base (UCI id=373):** leitura via `ucimlrepo` com fallback automático para o arquivo `drug+consumption+quantified.zip` caso não haja conexão com a internet.
2. **Estatísticas descritivas:** verificação de dimensões (`1.885 × 32`), tipos, valores ausentes (`0`), duplicatas (`0`), distribuição original de `cannabis` (`CL0` a `CL6`) e correlação entre traços de personalidade.
3. **Transformação nas colunas:** decodificação dos valores demográficos quantificados para rótulos legíveis, binarização do alvo (`CL0`/`CL1` = não usuário; `CL2`–`CL6` = usuário) e criação de `ss_impulsivo`.
4. **Limpeza nas linhas:** remoção de 8 respondentes que declararam uso de **Semeron** (droga fictícia usada para identificar respostas inconsistentes), totalizando **1.877 registros** e **13 variáveis preditoras**.
5. **Divisão estratificada:** **60% treino** (`1.125`), **20% validação** (`376`) e **20% teste** (`376`), preservando ~67% de usuários em todos os subconjuntos.
6. **Pipeline e treinamento:** `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`) + `RandomForestClassifier` (`n_estimators=300`, `class_weight="balanced"`), com ajuste de `max_depth` na validação pelo F1-score da classe usuário.
7. **Avaliação no teste:** relatório de classificação, ROC-AUC, acurácia e matriz de confusão.
8. **Implantação e predição:** serialização do pipeline completo em `artefatos/modelo_cannabis.joblib`, recarregamento do disco e predição em um caso real do teste e em um perfil novo simulado.

---

## Principais resultados (conjunto de teste — 376 amostras)

| Métrica | Resultado |
|---|---|
| Melhor `max_depth` (validação) | `livre` (`None`) — F1 val = `0,892` |
| **Acurácia (teste)** | **`78,19%`** |
| **ROC-AUC (teste)** | **`0,8519`** |
| **Usuário (classe 1)** — Precision / Recall / F1 | `0,82` / `0,87` / `0,84` |
| **Não usuário (classe 0)** — Precision / Recall / F1 | `0,69` / `0,61` / `0,65` |
| **Matriz de confusão** | `VN = 76` \| `FP = 48` \| `FN = 34` \| `VP = 218` |

---

## Como executar

### Pré-requisitos
- **Python 3.10 ou superior** (64-bit).
- Editor com suporte a Jupyter Notebooks (ex.: **VS Code** com a extensão *Jupyter*) ou terminal.

### Passo a passo (Windows / PowerShell)

1. **Entre na pasta do projeto:**
   ```powershell
   cd Machine-Learning-Drug-Consumption
   ```

2. **Crie e ative o ambiente virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instale as dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Execute o notebook:**
   - **No VS Code:** abra `esteira_classificacao_drug_consumption.ipynb`, selecione o kernel **`.venv`** no canto superior direito e clique em **Run All** (ou execute célula por célula com `Shift + Enter`).
   - **Pelo terminal:**
     ```powershell
     python -m jupyter nbconvert --to notebook --execute --inplace esteira_classificacao_drug_consumption.ipynb
     ```

---

## Referência

Fehrman, E., Egan, V., & Mirkes, E. (2015). *Drug Consumption (Quantified)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TC7S
