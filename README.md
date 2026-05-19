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
python -m pip check
```

Puis sélectionner le venv comme kernel dans Jupyter ou VSCode, et ouvrir un notebook.

Au premier lancement, les notebooks téléchargent les prix ETFs depuis Yahoo Finance et les taux depuis FRED, puis cachent tout dans `data/raw/`. Les exécutions suivantes lisent le cache, donc ça tourne hors ligne.

## Organisation du repo

```
README.md
requirements.txt
src/
  data.py          fetch yfinance + FRED, cache parquet
  backtest.py      signaux V1/V4, engine, métriques rf-aware
  robustness.py    sensibilité, bootstrap, PSR, Newey-West,
                   Jobson-Korkie, ADF, walk-forward, CAPM
data/raw/          cache des données (parquet)
notebooks/
  01_EDA.ipynb                      exploration des données, test ADF sur spread
  02_backtest_and_strategy.ipynb    stratégie + itérations V1 à V4, robustesse, walk-forward
  03_paper_trading_analysis.ipynb   paper trading 2024-2026 vs BH IEF
  FINAL_notebook.ipynb              version livrée, 12 sections
```

L'ordre logique de lecture : 01, puis 02, puis 03, puis FINAL. Le code partagé est dans `src/` ; les notebooks l'importent.

## Données utilisées

| Source | Série | Description |
|---|---|---|
| Yahoo Finance | TLT, IEF, SHY | Univers principal (Treasury ETFs iShares) |
| Yahoo Finance | AGG, SPY | Benchmarks élargis (Aggregate Bond + S&P 500) |
| FRED | DGS2, DGS10 | Taux US 2 ans et 10 ans constant maturity, base du signal |
| FRED | DGS3MO | T-Bill 3 mois, proxy du taux sans risque pour Sharpe ajusté |

Période : 2003-01 à aujourd'hui. Fréquence daily. Prix yfinance ajustés splits et dividendes. FRED via leur endpoint CSV public, pas besoin de clé API.

## Paramètres du backtest

| Élément | Valeur |
|---|---|
| Univers | TLT, IEF, SHY |
| Long-only | Oui, somme des poids = 1 |
| Rebalancing | Mensuel, dernier jour de **trading** du mois (calendrier ETF, pas calendaire) |
| Lag d'exécution | 1 jour de trading (jour de trading réel, pas BusinessDay civil) |
| Lag publication FRED | yields shiftés de 1 jour avant construction du signal |
| Transaction cost | 2 bps par côté |
| Slippage backtest | 2 bps |
| Slippage paper | 5 bps |

## Méthodologie et tests

Le projet intègre les contrôles méthodologiques standards d'un backtest sérieux :

| Contrôle | Implémentation |
|---|---|
| Look-ahead bias | Shift FRED + calendrier ETF + lag jours de trading réels |
| Stationnarité du signal | Test ADF sur spread 2s10s (non stationnaire, exploité en régimes) |
| Sensibilité paramètres | Heatmap 1D (seuils V1) et 2D (threshold_high × lookback V4) |
| Walk-forward | Refit `(low, high, lookback)` tous les 3 ans sur fenêtre train 5 ans glissante |
| Significance Sharpe | Block bootstrap + PSR + Newey-West + Jobson-Korkie |
| Benchmarks multiples | BH IEF, BH AGG, 60/40 SPY/IEF, alpha CAPM sur (AGG, SPY) |
| Sharpe rf | Calculé avec rf=0 et avec rf=DGS3MO |

## Résultats

Backtest 2003 à 2026 avec frais inclus :

| Version | CAGR | Vol | Sharpe | Calmar | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|---:|
| V1 sans coûts | 5.07% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V2 + transaction costs | 5.05% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V3 + slippage | 5.03% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V4 + filtre trend | 4.08% | 8.9% | 0.50 | 0.19 | -21.2% | 3.07 |
| Buy & Hold IEF | 3.38% | 6.8% | 0.52 | 0.14 | -23.9% | — |
| Buy & Hold AGG | 3.06% | 5.2% | 0.61 | 0.17 | -18.4% | — |
| 60/40 SPY/IEF | 8.72% | 10.7% | 0.83 | 0.28 | -31.4% | — |

Paper trading 2024-2026 avec V4 et exécution plus réaliste (lag T+2, slippage 5 bps) :

| Période | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| In-sample 2003-2023 | 4.39% | 9.3% | 0.51 | -21.2% |
| OOS backtest 2024-2026 | 1.30% | 3.9% | 0.35 | -3.95% |
| OOS paper 2024-2026 | 1.09% | 3.9% | 0.30 | -4.68% |
| BH IEF OOS 2024-2026 | 2.32% | 6.0% | 0.41 | -6.89% |

Tests de significance V4 vs BH IEF (full history) :

| Test | Résultat |
|---|---|
| Jobson-Korkie (différence de Sharpe) | z = −0.13, p = 0.90 |
| Newey-West (alpha vs IEF) | α = +0.87%/an, t = 0.66, p = 0.51 |
| Probabilistic Sharpe Ratio vs IEF | 45% |
| CAPM regression sur (AGG, SPY), rf=DGS3MO | α = +3.51%/an, t = 2.31, **p = 0.021** |
| Walk-forward (refit / 3 ans) vs paramètres a priori | Sharpe 0.48 vs 0.47 |
