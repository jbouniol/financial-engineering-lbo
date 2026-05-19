# Bond ETF Rotation via Yield Curve Regimes

Projet académique — module *Financial Engineering & Intro to Trading*.

Stratégie de rotation entre trois ETFs obligataires US (TLT / IEF / SHY) pilotée par le régime de la courbe des taux (spread 2s10s), avec overlay de risk management.

**Groupe :** Jonathan Bouniol, Guillaume RABEAU, Sacha NARDOUX, Florent Negaf, Enzo Natali.

---

## L'idée en 5 lignes

On allouait 100% du capital sur un seul ETF obligataire US à la fois (TLT long duration, IEF intermédiaire, SHY court), choisi mensuellement selon le régime de la courbe des taux US. Spread 2s10s pentu et positif → TLT (on prend du carry). Spread aplati → IEF (sensibilité plus mesurée). Spread inversé → SHY (refuge, on évite la duration). Une couche risk management (filtre trend 3 mois) bascule sur SHY si l'ETF sélectionné est en trend baissier — protège contre les régimes 2022-like où la courbe est en retard sur le prix.

## Structure du repo

```
.
├── README.md                              ← ce fichier
├── requirements.txt                       ← dépendances Python
├── .gitignore
├── Docs/
│   └── group_project_guidelines.md        ← consignes du prof
├── src/
│   ├── __init__.py
│   ├── data.py                            ← fetch yfinance + FRED, cache parquet
│   └── backtest.py                        ← signals V1/V4, engine, métriques
├── data/raw/
│   ├── etf_prices.parquet                 ← cache TLT/IEF/SHY (auto-adjusted)
│   └── fred_yields.parquet                ← cache DGS2 / DGS10
└── notebooks/
    ├── 01_EDA.ipynb                       ← exploration des données
    ├── 02_backtest_and_strategy.ipynb     ← stratégie + itérations V1→V4
    ├── 03_paper_trading_analysis.ipynb    ← OOS 2024-2026 + paper sim
    └── FINAL_notebook.ipynb               ← notebook livré (12 sections)
```

## Setup

```bash
# 1. Créer un venv (Python 3.12 recommandé)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Sélectionner ce venv comme kernel dans Jupyter / VSCode
```

Premier lancement : les notebooks téléchargent automatiquement les données depuis yfinance et FRED, puis les cachent dans `data/raw/`. Les exécutions suivantes lisent le cache (offline ok).

## Comment lire ce repo

L'ordre logique de lecture :

1. **`notebooks/01_EDA.ipynb`** — données, stats descriptives, premières intuitions de signal.
2. **`notebooks/02_backtest_and_strategy.ipynb`** — design de la stratégie, itérations V1 → V4, métriques, analyse des biais.
3. **`notebooks/03_paper_trading_analysis.ipynb`** — paper trading simulé sur 2024-2026, comparaison vs backtest.
4. **`notebooks/FINAL_notebook.ipynb`** — version consolidée, 12 sections, reproductible bout en bout.

Le code partagé entre notebooks est dans `src/` :
- `src/data.py` expose `load_all()` qui retourne les prix ETFs + yields FRED alignés.
- `src/backtest.py` expose les signals (`signal_v1`, `signal_v4`), l'engine (`run_backtest`), les métriques (`perf_metrics`), le trade log (`trade_log`) et le benchmark (`buy_and_hold`).

## Données

| Source | Série | Description |
|---|---|---|
| Yahoo Finance (`yfinance`) | TLT | iShares 20+ Year Treasury Bond ETF |
| Yahoo Finance | IEF | iShares 7-10 Year Treasury Bond ETF |
| Yahoo Finance | SHY | iShares 1-3 Year Treasury Bond ETF |
| FRED (`fredgraph.csv`) | DGS2 | US Treasury 2Y constant maturity yield (%) |
| FRED | DGS10 | US Treasury 10Y constant maturity yield (%) |

- Période : 2003-01-02 → date courante (les ETFs SHY/IEF démarrent en 2002-07, TLT idem).
- Fréquence : daily.
- Prix yfinance : `auto_adjust=True` → close ajusté splits + dividendes (cohérent backtest total return).
- FRED : endpoint CSV public, **pas de clé API requise**.

## Stratégie — résumé technique

| Élément | Choix |
|---|---|
| Univers | TLT, IEF, SHY (long-only, somme des poids = 1) |
| Fréquence | Rebalancing mensuel — dernier jour ouvré du mois |
| Lag d'exécution | 1 jour ouvré (close T → close T+1) → exclut le look-ahead |
| Signal V1 | Bucket sur le niveau du spread 2s10s : `>1%` → TLT, `0..1%` → IEF, `≤0%` → SHY |
| Signal V4 | V1 + filtre trend 3 mois (si ETF sélectionné en return négatif 3M → bascule SHY) |
| Transaction cost | 2 bps par côté (backtest), 2 bps (paper) |
| Slippage | 2 bps (backtest), 5 bps (paper) |
| Capital | Base index 1.0 |

## Résultats principaux

**Backtest 2003-01 → 2026-05 (full history, frais inclus)**

| | CAGR | Vol | Sharpe | MaxDD | Turnover/an |
|---|---:|---:|---:|---:|---:|
| V1 (sans coûts) | 5.08% | 11.0% | 0.51 | -26.6% | 0.92 |
| V2 (+ TC) | 5.06% | 11.0% | 0.50 | -26.6% | 0.92 |
| V3 (+ slippage) | 5.04% | 11.0% | 0.50 | -26.6% | 0.92 |
| **V4 (+ risk mgmt)** | 4.16% | 8.8% | 0.51 | **-21.2%** | 2.94 |
| Buy & Hold IEF | 3.37% | 6.8% | 0.52 | -23.9% | — |

**Paper trading 2024-01 → 2026-05 (out-of-sample, V4)**

| | CAGR | Vol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| IS backtest (2003-2023) | 4.47% | 9.2% | 0.52 | -21.2% |
| OOS backtest (2024-2026) | 1.30% | 3.8% | 0.36 | -3.95% |
| OOS paper (T+2, slip 5bps) | 0.99% | 3.9% | 0.27 | -4.68% |

L'écart IS ↔ OOS s'explique par l'absence de bascule de cycle franche sur 2024-2026 (régime macro défavorable au signal courbe). L'écart OOS backtest ↔ OOS paper s'explique par le slippage supplémentaire + le drift de latence T+2 (drift moyen absolu observé : ~22 bps par trade).

## Méthodologie — points clés

- **Pas de look-ahead** : signal calculé à la clôture du jour t, exécution à la clôture t+1 (`monthly_to_daily_weights(execution_lag=1)`).
- **Pas de data snooping** : seuils du signal (0% et 1%) choisis a priori sur base économique (inversion = signal de récession, prime de terme normale > 100 bps), pas optimisés sur la série.
- **Pas d'optimisation OOS** : le filtre trend V4 utilise un lookback 3M choisi a priori, le notebook 03 sert d'OOS naturel.
- **Costs intégrés** : transaction cost + slippage explicites dans toutes les versions V2+.
- **Reproductibilité** : seeds fixées, chemins relatifs, cache parquet, `requirements.txt` versionné.

## Livrables (selon brief)

| Livrable | Pondération | Fichier |
|---|---:|---|
| Notebook + datasets | 30% | `notebooks/FINAL_notebook.ipynb` + `data/raw/*.parquet` |
| Report 10-15 pages | 30% | à rédiger |
| Présentation 15 min + 5 Q&A | 40% | à préparer |

Deadline soumission Blackboard : **2026-05-31**. Présentation : **2026-05-29**.

## Limitations assumées

- Univers réduit à 3 ETFs Treasury — pas de TIPS, MBS, credit.
- Slippage modélisé en bps constants — sous-estime probablement les jours FOMC / NFP.
- Edge dépendant du cycle : sur les ~28 mois OOS sans bascule franche, Sharpe ~0.27.
- Paramètres du filtre V4 (lookback 3M) non walk-forward — choisis a priori mais non re-calibrés dans le temps.

## Améliorations futures

1. Régime-switching model (HMM 2-3 états) à la place des seuils en dur.
2. Sizing continu (sigmoïde sur le spread) plutôt que buckets discrets.
3. Univers élargi : TIP (TIPS), LQD (corporate IG).
4. Walk-forward des paramètres pour réduire le risque d'overfit.
5. Modèle de slippage state-dependent (jours macro élargis).
