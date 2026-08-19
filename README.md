# Analise de NDVI por Safra e DAS

Projeto em Python para analisar curvas de `NDVI` de um talhao a partir de um `KMZ` em `inputs/`, usando `Google Earth Engine` e `Sentinel-2`.

## O que o projeto faz

- `teste_ndvi.py`
  Gera curvas de NDVI por calendario para safras completas.

- `analise_das_ndvi.py`
  Gera curvas padronizadas por `DAS` a partir de datas estimadas de plantio e colheita definidas no topo do script.

## Estrutura esperada

```text
inputs/
  talhao.kmz

outputs/
  2018-2019/
  2019-2020/
  ...
```

## Requisitos

- Python 3.10+
- Conta autenticada no Google Earth Engine
- Projeto no Google Cloud habilitado para Earth Engine

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

### 1. Autentique o Earth Engine

```bash
source .venv/bin/activate
python3 -c "import ee; ee.Authenticate()"
```

### 2. Defina o projeto, se necessario

```bash
export EE_PROJECT_ID=ndvi-estudos
```

### 3. Execute a analise por DAS

```bash
python3 analise_das_ndvi.py
```

Ou abra o notebook:

```bash
jupyter notebook teste.ipynb
```

## Saidas da analise por DAS

Para cada safra em `outputs/<safra>/`:

- `ndvi_das_<safra>.csv`
- `curva_ndvi_das_<safra>.png`
- `mapas_das/` com uma imagem por DAS

No final da execucao:

- `outputs/ndvi_das_consolidado.csv`
- `outputs/curvas_ndvi_das_safras_sobrepostas.png`

## Ajuste das safras

Edite a lista `SAFRAS_PADRONIZADAS` no topo de `analise_das_ndvi.py`:

```python
SAFRAS_PADRONIZADAS = [
    {"safra": "2019-2020", "inicio": "2019-11-12", "fim": "2020-03-31"},
]
```

## Observacoes

- O talhao deve estar em `inputs/` em formato `KMZ`.
- O script usa `NDVI mediano do talhao`.
- Falhas de leitura podem ocorrer por nuvem, sombra ou ausencia de imagens validas no periodo.
