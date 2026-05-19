# Bond ETF Rotation via Yield Curve Regimes

Projet du module *Financial Engineering & Intro to Trading*.

Groupe : Jonathan Bouniol, Guillaume RABEAU, Sacha NARDOUX, Florent Negaf, Enzo Natali.

## Ce que fait la stratégie

Chaque mois, on place 100% du capital sur un seul ETF obligataire américain parmi trois : TLT (obligations longues, 20+ ans), IEF (intermédiaires, 7 à 10 ans), SHY (courtes, 1 à 3 ans). Le choix dépend de la forme de la courbe des taux US, mesurée par le spread entre le taux 10 ans et le taux 2 ans (DGS10 moins DGS2).

Quand le spread est largement positif (> 1%), la courbe est normale et pentue : on prend TLT pour capter le carry sur la duration longue. Quand le spread se resserre (entre 0 et 1%), on passe sur IEF, moins exposé. Quand la courbe s'inverse (spread négatif), historiquement un signal de récession qui arrive, on se réfugie sur SHY pour éviter de tenir de la duration pendant que la Fed monte les taux.

Une deuxième couche regarde la performance des 3 derniers mois de l'ETF choisi par la règle. Si elle est négative, on bascule sur SHY le temps que le prix se stabilise. C'est ce qui rattrape la stratégie sur 2022, où la courbe restait positive pendant que TLT s'effondrait.

## Comment faire tourner le projet

Python 3.12 recommandé.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Puis sélectionner le venv comme kernel dans Jupyter ou VSCode, et ouvrir un notebook.

Au premier lancement, les notebooks téléchargent les prix ETFs depuis Yahoo Finance et les taux depuis FRED, puis cachent tout dans `data/raw/`. Les exécutions suivantes lisent le cache, donc ça tourne hors ligne.

## Organisation du repo

```
README.md
requirements.txt
src/
  data.py        fetch yfinance + FRED, cache parquet
  backtest.py    signaux V1/V4, engine, métriques, trade log
data/raw/        cache des données (parquet)
notebooks/
  01_EDA.ipynb                      exploration des données
  02_backtest_and_strategy.ipynb    stratégie + itérations V1 à V4
  03_paper_trading_analysis.ipynb   paper trading 2024-2026
  FINAL_notebook.ipynb              version livrée, 12 sections
```

L'ordre logique de lecture : 01, puis 02, puis 03, puis FINAL. Le code partagé est dans `src/` ; les notebooks l'importent.

## Données utilisées

| Source | Série | Description |
|---|---|---|
| Yahoo Finance | TLT | iShares 20+ Year Treasury Bond ETF |
| Yahoo Finance | IEF | iShares 7-10 Year Treasury Bond ETF |
| Yahoo Finance | SHY | iShares 1-3 Year Treasury Bond ETF |
| FRED | DGS2 | Taux US 2 ans constant maturity |
| FRED | DGS10 | Taux US 10 ans constant maturity |

Période : 2003-01 à aujourd'hui. Fréquence daily. Prix yfinance ajustés splits et dividendes. FRED via leur endpoint CSV public, pas besoin de clé API.

## Paramètres du backtest

| Élément | Valeur |
|---|---|
| Univers | TLT, IEF, SHY |
| Long-only | Oui, somme des poids = 1 |
| Rebalancing | Mensuel, dernier jour ouvré du mois |
| Lag d'exécution | 1 jour ouvré (évite tout look-ahead) |
| Transaction cost | 2 bps par côté |
| Slippage backtest | 2 bps |
| Slippage paper | 5 bps |

## Résultats

Backtest 2003 à 2026 avec frais inclus, comparé à un buy and hold IEF :

| Version | CAGR | Vol | Sharpe | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|
| V1 sans coûts | 5.08% | 11.0% | 0.51 | -26.6% | 0.92 |
| V2 + transaction costs | 5.06% | 11.0% | 0.50 | -26.6% | 0.92 |
| V3 + slippage | 5.04% | 11.0% | 0.50 | -26.6% | 0.92 |
| V4 + filtre trend | 4.16% | 8.8% | 0.51 | -21.2% | 2.94 |
| Buy & Hold IEF | 3.37% | 6.8% | 0.52 | -23.9% | — |

Paper trading 2024-2026 avec V4 et exécution plus réaliste (lag T+2, slippage 5 bps) :

| Période | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| In-sample 2003-2023 | 4.47% | 9.2% | 0.52 | -21.2% |
| OOS backtest 2024-2026 | 1.30% | 3.8% | 0.36 | -3.9% |
| OOS paper 2024-2026 | 0.99% | 3.9% | 0.27 | -4.7% |
