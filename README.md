# Filtering Before Ranking: Evaluating LLM-Extracted Metadata Constraints for Dense Retrieval in Fashion E-Commerce

Este repositório contém a arquitetura completa de código, os prompts expandidos, os vocabulários fechados, os scripts de auditoria pós-hoc, as figuras oficiais e as tabelas consolidadas do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação da **Universidade Federal do Rio Grande do Sul (UFRGS)**:

* **Autora:** Raíssa Bitencourt Borowski
* **Orientador:** Prof. Dr. Joel Luís Carbonera
* **Ano:** 2026

---

## Arquitetura do Repositório

```text
tcc-hybrid-fashion-search/
│
├── README.md                                  # Documentação principal, instruções de reprodução e guia dos artefatos
├── .gitignore                                 # Regras para ignorar caches pesados (.npy, articles.csv bruto)
├── requirements_observed.txt                  # Versões exatas dos pacotes observadas na execução congelada (Colab T4)
│
├── notebooks/                                 # Notebooks Jupyter (.ipynb) organizados cronologicamente
│   ├── 00_development_pilot_20_queries.ipynb  # Estudo piloto exploratório inicial (20 consultas sintéticas e casos manuais)
│   ├── 01_experiment_pipeline.ipynb           # Pipeline principal completo de validação (10.000 itens, 200 queries sintéticas, 40 de atributos)
│   └── 02_post_hoc_audits.ipynb               # Todas as auditorias pós-hoc, testes de McNemar, bootstrap agrupado e verificação de dados
│
├── src/                                       # Módulos Python (.py) extraídos para execução direta e inspeção de código
│   ├── tcc_validacao_worker.py                # Motor principal de execução isolada por subprocessos (prepare, generate, parse, minilm, bge, report)
│   ├── post_hoc_audits.py                     # Script Python completo que reproduz todas as análises e contagens do Capítulo 5 e Apêndices
│   └── generate_figures.py                    # Script para regenerar os gráficos de curvas Hit@k e intervalos de confiança pareados
│
├── materials/                                 # Especificações congeladas do desenho experimental (Apêndices A, B e D)
│   ├── parser_prompt.json                     # Prompt exato do sistema, esquema JSON e os 4 exemplos few-shot enviados ao Qwen2.5-1.5B-Instruct
│   ├── vocabulary.json                        # Vocabulário fechado permitido (74 tipos de produto, 13 famílias de cor e 5 grupos de público)
│   ├── normalization_rules.json               # Tabela determinística de sinônimos (SYNONYMS) antes da validação no vocabulário
│   ├── colour_mapping_audit.csv               # Tabela completa de mapeamento das cores brutas da H&M para as 13 famílias de cor
│   ├── environment.json                       # Metadados do ambiente de execução (GPU NVIDIA Tesla T4, seeds 42 e 2026, políticas ativas)
│   └── validation_design.json                 # IDs dos 20 alvos do piloto excluídos, product_codes bloqueados e tamanho do pool elegível (8.197)
│
├── data_and_results/                          # Tabelas de resultados da execução de validação (e local para extrair resultados_validacao.zip)
│   ├── summary.csv                            # Tabela 5.3: Hit@1, Hit@5, Hit@10, Hit@20, MRR@20, exclusões e tamanho médio do conjunto candidato
│   ├── paired_intervals.csv                   # Tabelas 5.4 e 5.5: Diferenças pareadas vs. baseline e IC 95% via bootstrap (5.000 reamostragens)
│   ├── paired_transitions_table_5_6.csv       # Tabela 5.6: Transições pareadas no Top-5 (Gained, Lost, Unchanged) e p-valores exatos de McNemar
│   ├── extraction_PROPOSED_LABELS.csv         # Tabela 5.8: Precisão, revocação, TP, FP e FN por campo nas 40 consultas controladas de atributos
│   └── label_space_summary.csv                # Tabela 5.1: Estatísticas do espaço de rótulos no catálogo H&M (105.542 itens) e na amostra (10.000 itens)
│   └── [Adicione aqui os CSVs/JSONs brutos extraídos do seu resultados_validacao.zip local: runs.csv, catalog_sample.csv, synthetic_queries.csv, etc.]
│
├── audits/                                    # Planilhas das auditorias manuais e revisões independentes (Capítulo 5 e Apêndices B e C)
│   ├── audit_13_exclusions.csv                # Tabela C.1 completa: Auditoria caso a caso dos 13 alvos excluídos pela filtragem (causa e confiança)
│   └── attribute_labels_reviewed.csv          # Apêndice B: As 40 consultas controladas de atributos revisadas por segundo anotador independente (100% acordo)
│   └── [O script 02_post_hoc_audits.ipynb também gera automaticamente audit_91_deletions.csv aqui ao ler runs.csv]
│
└── figures/                                   # Todas as figuras e painéis visuais em alta resolução da monografia (Figuras 5.1 a 5.7)
    ├── fig_5_1_recovery_curves.png
    ├── fig_5_2_paired_differences_intervals.png
    ├── fig_5_3_constraint_violations.png
    ├── fig_extra_candidate_distribution.png
    ├── fig_5_4_case1_minilm_improvement_synth_0165.png
    ├── fig_5_5_case2_bge_deterioration_synth_0161.png
    ├── fig_5_6_case3_bge_exclusion_jacket_blazer_synth_0004.png
    └── fig_5_7_case4_minilm_compliance_white_sneakers_attr_000.png
```

---

## Como Adicionar os Arquivos Brutos do Colab (`resultados_validacao.zip`)

Ao baixar o arquivo `resultados_validacao.zip` gerado pelo Google Colab no seu computador:
1. Você pode manter o próprio `resultados_validacao.zip` na raiz deste repositório **E/OU** extrair o conteúdo dele dentro da pasta `data_and_results/`:
   * `runs.csv` (as 1.200 linhas de avaliação das 200 consultas sintéticas × 3 sistemas × 2 encoders)
   * `catalog_sample.csv` (os 10.000 artigos amostrados com `SEED = 42`)
   * `validation_pool.csv` (os 8.197 artigos únicos elegíveis após exclusão do piloto e deduplicação)
   * `synthetic_queries.csv` (os 200 pares consulta–artigo gerados pelo Qwen2.5-3B-Instruct com `VALIDATION_SEED = 2026`)
   * `attribute_cases_PROPOSED_LABELS.csv` (avaliação por consulta das 40 queries controladas)
   * `parser_<hash>.json` e `parser_attribute_outputs.json` (respostas brutas em cache do parser Qwen2.5-1.5B-Instruct)
   * `synthetic_queries_to_review.csv` / `audit_50_sample.csv` (planilha da auditoria de plausibilidade das 50 consultas)

---

## Como Reproduzir as Análises

### 1. Reproduzir Tabelas e Auditorias Pós-Hoc (Sem GPU / Instantâneo)
Com `resultados_validacao.zip` na raiz (ou descompactado em `data_and_results/`), execute o notebook `notebooks/02_post_hoc_audits.ipynb` ou rode diretamente no terminal:
```bash
python src/post_hoc_audits.py
```
Esse script verifica automaticamente:
* O truncamento de tokens nos 10.000 itens (0% em 256 WordPieces no MiniLM e 0% em 512 tokens no BGE);
* Os 26 tipos raros (< 5 ocorrências = 52 artigos) e `Unknown` (13 artigos);
* A separação estrita dos 20 itens do piloto (32 artigos removidos) e a deduplicação para 8.197 alvos elegíveis;
* A ausência de cores nas 200 consultas sintéticas (0/200) e os 2 pares de consultas com texto idêntico (198 textos únicos);
* A Tabela 5.6 de transições pareadas (*Gained*, *Lost*, *Unchanged*), os p-valores do **teste exato de McNemar** e o **bootstrap agrupado pós-hoc** (10.000 reamostragens, `seed = 20261017`);
* A decomposição das 200 consultas residuais (101 sem filtro, 8 multi-palavra/fallback e 91 deleções efetivas de uma palavra) e o desempenho estratificado nas 91 consultas alteradas;
* A auditoria de falhas estruturais (2/200), descartes de validação (94/200) e chaves JSON duplicadas (11/200 sintéticas, 1/40 atributos).

### 2. Reproduzir o Experimento Completo do Zero (Requer GPU NVIDIA T4 no Google Colab)
1. Abra `notebooks/01_experiment_pipeline.ipynb` no Google Colab com acelerador **GPU T4**.
2. Disponibilize `articles.csv` do dataset *H&M Personalized Fashion Recommendations* (ou configure `KAGGLE_API_TOKEN` nos Secrets do Colab).
3. Execute as células sequencialmente. O controlador executa as 6 etapas (`prepare`, `generate`, `parse`, `minilm`, `bge`, `report`) em subprocessos independentes para garantir isolamento de memória RAM/VRAM e gera `resultados_validacao.zip`.

> **Nota de Auditoria sobre o Template de Texto do Produto (§4.3):** Na função `load_catalog()` de `src/tcc_validacao_worker.py`, a string concatenada para indexação densa utiliza o prefixo literal `'Audience: '` antes do valor de `index_group_name` (conforme preservado nos artefatos congelados da execução oficial). Todos os três sistemas avaliados (`baseline`, `filters_full` e `filters_soft`) compartilham exatamente os mesmos vetores de catálogo pré-computados a partir dessa representação textual idêntica.
