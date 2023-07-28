<dic=v style="text-align: center;"># Previsão de Over 0.5 HT<div>

![lars-bo-nielsen-Wu7hYE7Lzzs-unsplash](https://github.com/VladeMelo/over-0.5-ht/assets/63476377/417fecce-edc6-499c-ae67-87dd35e4c878)

## 1. Objetivo

Tentar prever se vai sair pelo menos um gol ainda no 1° tempo de uma partida que está 0 x 0 aos 18 minutos de jogo.

## 2. Tratamento dos Dados

- Altera algumas colunas do tipo Objeto para o tipo Numérico
- Remove partidas em que não houveram ações
- Remove as colunas Game, League e Winner
- Remove possíveis Outliers
- Utiliza RandomUnderSampler para balancear os dados em relação a coluna Over 0.5 HT
- Separa os dados entre: 70% treino e 30% teste

## 3. Algoritmo para criação do Modelo

GradientBoostingClassifier por motivos de estudos pessoais, mas futuramente haverá adição de novos modelos.

## 4. Avaliação dos melhores parâmetros

- Constante IQR para a análise de Outliers:

| Parâmetro  | Valores |
| ------------- | ------------- |
| IQR_CONST  | 1, 1.25, 1.5, 1.75, 2  |

- Hiperparâmetros do GradientBoostingClassifier:

| Parâmetro  | Valores |
| ------------- | ------------- |
| learning_rate  | 0.05, 0.15, 0.25, 0.35, 0.45 |
| subsample  | 0.1, 0.3, 0.5, 0.7, 0.9 |
| n_estimators  | 1, 2, 4, 8, 16, 32, 64, 100, 200 |
| max_depth  | 1, 2, 3, 4 |

- Também foi aplicado o método de Validação Cruzada K-Fold com o K sendo 5 para uma melhor precisão

## 5. Resultados

- Parâmetros que melhor performaram:

| Parâmetro  | Valor |
| ------------- | ------------- |
| IQR_CONST  | 1.5 |
| learning_rate  | 0.45 |
| subsample  | 0.5 |
| n_estimators  | 64 |
| max_depth  | 4 |

- Resultados utilizando os parâmetros acima:

| Métrica  | Valor |
| ------------- | ------------- |
| Acurácia Média  | 57,32% |
| Acurácia Mediana  | 59,75% |
| Recall para o valor 0  | 57,73% |
| Recall para o valor 1  | 58,42% |

## 6. Conclusão

![pie_chart](https://github.com/VladeMelo/over-0.5-ht/assets/63476377/275c0dbd-1b59-4dcd-993f-133b58c008a8)

- Levando em consideração a distribuição dos valores da feature Over 0.5 HT, podemos dizer que o modelo teve uma boa performance!

| Valor do Over 0.5 HT  | Upside |
| ------------- | ------------- |
| 0  | 5,07% |
| 1  | 26,11% |

## Próximos Passos / Em Andamento

- Avaliar com outros modelos
- Analisar possíveis casos de Multicolinearidade entre algumas features
- Treinar o modelo em apenas algumas ligas (Ex: Brasileirão Série A e Série B)
