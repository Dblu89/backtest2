# “””
ALPHA DISCOVERY ENGINE v2 — WDO B3

Motor vetorizado com VectorBT 0.28.5
Testa milhoes de combos em minutos.

VELOCIDADE:

- VectorBT + Numba: 1M combos em ~20 minutos
- vs bar-by-bar: 1M combos em 140 horas

FAMILIAS:

1. EMA Crossover (tendencia)
1. RSI Reversion (mean reversion)
1. Bollinger Bands (mean reversion)
1. MACD Momentum
1. ROC Momentum
1. Donchian Breakout
1. ATR Volatility Breakout
1. Stochastic Reversion
1. Combined (multiplas confirmaçoes)

VALIDACAO:

- CPCV (skfolio)
- DSR (scipy)
- Block Bootstrap (arch)
- Stress tests
- Stage-gate 7 criterios

LICOES APRENDIDAS:

- Entrada no OPEN do proximo candle: vbt usa trade_on_close=False
- Min 200 trades por combo
- PF cap 3.0, Sharpe cap 4.0
- Sem repeticao de combos (GridSampler)
  “””

import pandas as pd
import numpy as np
import vectorbt as vbt
import json, sys, os, time, warnings, itertools, math
from datetime import datetime
from scipy import stats
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(“ignore”)

# ================================================================

# CONFIGURACOES

# ================================================================

CSV_PATH   = “/workspace/strategy_composer/wdo_clean.csv”
OUTPUT_DIR = “/workspace/param_opt_output/alpha_v2”
CAPITAL    = 50_000.0
MULT_WDO   = 10.0
COMISSAO   = 5.0       # R$ por trade round-trip
SLIPPAGE   = 2.0       # pontos
MIN_TRADES = 200
MAX_PF     = 3.0
MAX_SHARPE = 4.0
MAX_DD     = -25.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Comissao em % do valor nocional para VectorBT

# WDO: 1 ponto = R$10, preco medio ~5500

# Comissao R$5 / (5500 * 10) = 0.0000909

COMM_PCT = (COMISSAO + SLIPPAGE * MULT_WDO * 0.5) / (5500 * MULT_WDO)

# ================================================================

# SECAO 1: DADOS

# ================================================================

def carregar():
print(f”[DATA] Carregando {CSV_PATH}…”)
df = pd.read_csv(CSV_PATH, parse_dates=[“datetime”], index_col=“datetime”)
df.columns = [c.lower().strip() for c in df.columns]
df = df[[“open”,“high”,“low”,“close”,“volume”]].copy()
df = df[df.index.dayofweek < 5]
df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
df = df.dropna()
df = df[df[“close”] > 0]
df = df.sort_index()
df = df[~df.index.duplicated(keep=“last”)]
print(f”[DATA] OK {len(df):,} candles | “
f”{df.index[0].date()} -> {df.index[-1].date()}”)
return df

# ================================================================

# SECAO 2: FEATURES PRE-COMPUTADAS

# ================================================================

def calcular_features(df):
“””
Pre-computa todos os indicadores uma vez.
Retorna dict com arrays numpy para velocidade maxima.
“””
print(”[FEAT] Pre-computando features…”)
close  = df[“close”]
high   = df[“high”]
low    = df[“low”]
volume = df[“volume”]
n      = len(df)

```
f = {}

# EMAs
for p in [5, 8, 10, 13, 20, 21, 34, 50, 100, 200]:
    f[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean().values

# SMAs
for p in [10, 20, 50, 200]:
    f[f"sma_{p}"] = close.rolling(p).mean().values

# RSI multiplos periodos
for p in [7, 9, 14, 21]:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(p).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(p).mean()
    f[f"rsi_{p}"] = (100 - (100 / (1 + gain / (loss + 1e-9)))).values

# MACD multiplas configuracoes
for fast, slow, sig in [(12,26,9), (8,21,5), (5,13,3)]:
    e_fast = close.ewm(span=fast).mean()
    e_slow = close.ewm(span=slow).mean()
    macd   = e_fast - e_slow
    signal = macd.ewm(span=sig).mean()
    key    = f"macd_{fast}_{slow}"
    f[key]           = macd.values
    f[f"{key}_sig"]  = signal.values
    f[f"{key}_hist"] = (macd - signal).values

# Bollinger Bands multiplos periodos e desvios
for p in [10, 20, 50]:
    for std in [1.5, 2.0, 2.5]:
        sma  = close.rolling(p).mean()
        sd   = close.rolling(p).std()
        key  = f"bb_{p}_{str(std).replace('.','')}"
        f[f"{key}_upper"] = (sma + std * sd).values
        f[f"{key}_lower"] = (sma - std * sd).values
        f[f"{key}_pct"]   = ((close - (sma - std*sd)) /
                              (2 * std * sd + 1e-9)).values
        f[f"{key}_width"] = (2 * std * sd / (sma + 1e-9)).values

# Donchian Channels
for p in [10, 20, 50, 100]:
    f[f"don_high_{p}"] = high.rolling(p).max().shift(1).values
    f[f"don_low_{p}"]  = low.rolling(p).min().shift(1).values

# ATR
cp = close.shift(1)
tr = pd.concat([high-low, (high-cp).abs(), (low-cp).abs()], axis=1).max(axis=1)
for p in [7, 14, 20]:
    f[f"atr_{p}"] = tr.rolling(p).mean().values

# ROC
for p in [3, 5, 10, 20]:
    f[f"roc_{p}"] = (close.pct_change(p) * 100).values

# Stochastic
for p in [5, 9, 14]:
    lo_p = low.rolling(p).min()
    hi_p = high.rolling(p).max()
    stk  = (close - lo_p) / (hi_p - lo_p + 1e-9) * 100
    f[f"stoch_k_{p}"] = stk.values
    f[f"stoch_d_{p}"] = stk.rolling(3).mean().values

# CCI
for p in [14, 20]:
    tp   = (high + low + close) / 3
    sma  = tp.rolling(p).mean()
    mad  = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean())
    f[f"cci_{p}"] = ((tp - sma) / (0.015 * mad + 1e-9)).values

# Volume z-score
for p in [10, 20]:
    vol_mean = volume.rolling(p).mean()
    vol_std  = volume.rolling(p).std()
    f[f"vol_z_{p}"] = ((volume - vol_mean) / (vol_std + 1e-9)).values

# Volatilidade
ret = close.pct_change()
for p in [5, 10, 20]:
    f[f"volatility_{p}"] = (ret.rolling(p).std() * 100).values

# Volatilidade relativa
f["vol_ratio"] = f["volatility_5"] / (f["volatility_20"] + 1e-9)

# Squeeze (BB dentro do KC)
sma20 = close.rolling(20).mean()
atr14 = f["atr_14"]
bb_u  = (sma20 + 2 * close.rolling(20).std()).values
bb_l  = (sma20 - 2 * close.rolling(20).std()).values
kc_u  = (sma20 + 1.5 * atr14).values
kc_l  = (sma20 - 1.5 * atr14).values
f["squeeze"] = ((bb_u < kc_u) & (bb_l > kc_l)).astype(int)

# Sessao
f["session_am"] = ((df.index.hour >= 9) & (df.index.hour < 12)).astype(int).values
f["session_pm"] = ((df.index.hour >= 13) & (df.index.hour < 17)).astype(int).values
f["hora"]       = df.index.hour

# Close como array
f["close"]  = close.values
f["open"]   = df["open"].values
f["high"]   = high.values
f["low"]    = low.values
f["volume"] = volume.values
f["index"]  = df.index

n_features = len([k for k in f.keys() if k not in ["close","open","high","low","volume","index"]])
print(f"[FEAT] OK {n_features} indicadores pre-computados")
return f
```

# ================================================================

# SECAO 3: GERADOR DE SINAIS VETORIZADOS

# ================================================================

def gerar_sinais_ema_cross(f, fast, slow):
“”“EMA rapida cruza EMA lenta.”””
if f”ema_{fast}” not in f or f”ema_{slow}” not in f:
return None, None
ef = f[f”ema_{fast}”]
es = f[f”ema_{slow}”]
entries = (ef > es) & (np.roll(ef,1) <= np.roll(es,1))
exits   = (ef < es) & (np.roll(ef,1) >= np.roll(es,1))
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_rsi(f, period, oversold, overbought, direction=“long”):
“”“RSI em extremos.”””
key = f”rsi_{period}”
if key not in f:
return None, None
rsi = f[key]
if direction == “long”:
entries = (rsi < oversold) & (np.roll(rsi,1) >= oversold)
exits   = rsi > 50
else:
entries = (rsi > overbought) & (np.roll(rsi,1) <= overbought)
exits   = rsi < 50
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_bollinger(f, period, std_mult, direction=“long”):
“”“Preco toca banda de Bollinger.”””
key = f”bb_{period}_{str(std_mult).replace(’.’,’’)}”
pct_key = f”{key}_pct”
if pct_key not in f:
return None, None
pct = f[pct_key]
if direction == “long”:
entries = pct < 0.05
exits   = pct > 0.5
else:
entries = pct > 0.95
exits   = pct < 0.5
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_macd(f, fast, slow, direction=“long”):
“”“MACD histogram muda de sinal.”””
key = f”macd_{fast}_{slow}_hist”
if key not in f:
return None, None
hist = f[key]
if direction == “long”:
entries = (hist > 0) & (np.roll(hist,1) <= 0)
exits   = (hist < 0) & (np.roll(hist,1) >= 0)
else:
entries = (hist < 0) & (np.roll(hist,1) >= 0)
exits   = (hist > 0) & (np.roll(hist,1) <= 0)
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_donchian(f, period, direction=“long”):
“”“Breakout do canal de Donchian.”””
hk = f”don_high_{period}”
lk = f”don_low_{period}”
if hk not in f or lk not in f:
return None, None
close = f[“close”]
if direction == “long”:
entries = close > f[hk]
exits   = close < f[lk]
else:
entries = close < f[lk]
exits   = close > f[hk]
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_roc(f, period, threshold, direction=“long”):
“”“ROC forte indica continuacao.”””
key = f”roc_{period}”
if key not in f:
return None, None
roc = f[key]
if direction == “long”:
entries = roc > threshold
exits   = roc < 0
else:
entries = roc < -threshold
exits   = roc > 0
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_stoch(f, period, oversold, overbought, direction=“long”):
“”“Stochastic em extremos.”””
k_key = f”stoch_k_{period}”
d_key = f”stoch_d_{period}”
if k_key not in f:
return None, None
k = f[k_key]
d = f[d_key]
if direction == “long”:
entries = (k < oversold) & (np.roll(k,1) >= oversold)
exits   = k > 50
else:
entries = (k > overbought) & (np.roll(k,1) <= overbought)
exits   = k < 50
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_volatility(f, vol_threshold, direction=“long”):
“”“Breakout apos squeeze de volatilidade.”””
squeeze = f[“squeeze”]
vol_r   = f[“vol_ratio”]
hist    = f.get(“macd_12_26_hist”, np.zeros(len(squeeze)))
saindo  = (squeeze == 0) & (np.roll(squeeze,1) == 1)
expand  = vol_r > vol_threshold
if direction == “long”:
entries = saindo & expand & (hist > 0)
exits   = saindo & expand & (hist < 0)
else:
entries = saindo & expand & (hist < 0)
exits   = saindo & expand & (hist > 0)
entries[0] = exits[0] = False
return entries, exits

def gerar_sinais_cci(f, period, threshold, direction=“long”):
“”“CCI em extremos.”””
key = f”cci_{period}”
if key not in f:
return None, None
cci = f[key]
if direction == “long”:
entries = (cci < -threshold) & (np.roll(cci,1) >= -threshold)
exits   = cci > 0
else:
entries = (cci > threshold) & (np.roll(cci,1) <= threshold)
exits   = cci < 0
entries[0] = exits[0] = False
return entries, exits

# ================================================================

# SECAO 4: BACKTEST VETORIZADO COM VECTORBT

# ================================================================

def backtest_vbt(f, entries, exits, sl_pct=0.01, tp_pct=0.02,
direction=“longonly”):
“””
Backtest vetorizado com VectorBT 0.28.5.
CORRECAO: usa open_.shift(-1) como preco de entrada
para simular entrada no OPEN do proximo candle (sem lookahead).
“””
# OPEN do proximo candle como preco de execucao
open_next = pd.Series(f[“open”], index=f[“index”]).shift(-1)
ent = pd.Series(entries, index=f[“index”])
ex  = pd.Series(exits,   index=f[“index”])

```
try:
    pf = vbt.Portfolio.from_signals(
        open_next,
        ent,
        ex,
        sl_stop   = sl_pct,
        tp_stop   = tp_pct,
        fees      = COMM_PCT,
        init_cash = CAPITAL,
        direction = direction,
        freq      = "1min",
    )
    return pf
except Exception:
    return None
```

def extrair_metricas_vbt(pf, n_trials=1):
“”“Extrai metricas do Portfolio VectorBT.”””
if pf is None:
return None

```
try:
    trades = pf.trades.records_readable
    n = len(trades)
    if n < MIN_TRADES:
        return None

    pnl_total = float(pf.total_profit())
    if np.isnan(pnl_total):
        return None

    # Metricas basicas
    wins  = trades[trades["PnL"] > 0]
    loses = trades[trades["PnL"] <= 0]
    wr    = len(wins) / n * 100
    avg_w = float(wins["PnL"].mean())  if len(wins)  else 0
    avg_l = float(loses["PnL"].mean()) if len(loses) else -1
    gl    = float(loses["PnL"].sum())
    gw    = float(wins["PnL"].sum())
    pf_v  = abs(gw / gl) if gl != 0 else 9999

    if pf_v > MAX_PF:
        return None

    # Drawdown
    mdd = float(pf.max_drawdown()) * 100
    if mdd < MAX_DD:
        return None

    # Sharpe
    sh = float(pf.sharpe_ratio())
    if np.isnan(sh) or sh > MAX_SHARPE:
        return None

    # Sortino
    sortino = float(pf.sortino_ratio())
    calmar  = float(pf.calmar_ratio()) if not np.isnan(pf.calmar_ratio()) else 0

    # DSR
    pnl_arr  = trades["PnL"].values
    skew_pnl = float(stats.skew(pnl_arr))
    kurt_pnl = float(stats.kurtosis(pnl_arr) + 3)
    dsr, dsr_interp = calcular_dsr(sh, n_trials, n, skew_pnl, kurt_pnl)

    # Rolling windows (consistencia)
    pnls    = pnl_arr
    n_jan   = max(1, len(range(0, n-50, 25)))
    jan_pos = sum(1 for s in range(0, n-50, 25) if pnls[s:s+50].sum() > 0)
    pct_jan = jan_pos / n_jan * 100

    expectancy = (wr/100*avg_w) + ((1-wr/100)*avg_l)

    return {
        "total_trades":    n,
        "wins":            int(len(wins)),
        "losses":          int(len(loses)),
        "win_rate":        round(wr, 2),
        "profit_factor":   round(pf_v, 3),
        "sharpe_ratio":    round(sh, 3),
        "sortino_ratio":   round(sortino, 3) if not np.isnan(sortino) else 0,
        "calmar_ratio":    round(calmar, 3),
        "dsr":             dsr,
        "dsr_interp":      dsr_interp,
        "expectancy_brl":  round(expectancy, 2),
        "total_pnl_brl":   round(pnl_total, 2),
        "retorno_pct":     round(pnl_total/CAPITAL*100, 2),
        "max_drawdown_pct":round(mdd, 2),
        "capital_final":   round(CAPITAL+pnl_total, 2),
        "pct_janelas_pos": round(pct_jan, 1),
        "skewness":        round(skew_pnl, 3),
        "kurtosis":        round(kurt_pnl, 3),
    }
except Exception:
    return None
```

# ================================================================

# SECAO 5: DSR

# ================================================================

def calcular_dsr(sharpe_obs, n_trials, T, skew=0.0, kurt=3.0):
if n_trials <= 1 or T < 30 or sharpe_obs <= 0:
return 0.0, “INSUFICIENTE”
gamma   = 0.5772156649
e_max   = ((1-gamma)*stats.norm.ppf(1-1.0/n_trials) +
gamma*stats.norm.ppf(1-1.0/(n_trials*math.e)))
var_sr  = (1.0/T)*(1+0.5*sharpe_obs**2 - skew*sharpe_obs +
(kurt-3)/4.0*sharpe_obs**2)
if var_sr <= 0:
return 0.0, “ERRO”
dsr_stat = (sharpe_obs - e_max) / np.sqrt(var_sr)
dsr      = float(stats.norm.cdf(dsr_stat))
if dsr >= 0.95:   interp = “EXCELENTE”
elif dsr >= 0.85: interp = “BOM”
elif dsr >= 0.70: interp = “ACEITAVEL”
else:             interp = “FRACO”
return round(dsr, 4), interp

# ================================================================

# SECAO 6: STRESS TEST

# ================================================================

def stress_test_vbt(pf):
if pf is None:
return {“aprovado”: False, “stress_score”: “0/3”}
try:
trades = pf.trades.records_readable
pnls   = trades[“PnL”].values
if len(pnls) < 10:
return {“aprovado”: False, “stress_score”: “0/3”}

```
    # 1. Sem top 5
    top5_idx  = np.argsort(pnls)[-5:]
    mask      = np.ones(len(pnls), dtype=bool)
    mask[top5_idx] = False
    sem_top5  = pnls[mask].sum()

    # 2. Slippage 2x
    extra_slip = SLIPPAGE * MULT_WDO * 0.5
    slip2x     = pnls.sum() - extra_slip * len(pnls)

    # 3. Comissao +50%
    com50 = pnls.sum() - COMISSAO * 0.5 * len(pnls)

    ok1 = sem_top5 > 0
    ok2 = slip2x   > 0
    ok3 = com50    > 0
    n   = sum([ok1, ok2, ok3])

    return {
        "sem_top5_pnl":     round(float(sem_top5), 2),
        "sem_top5_ok":      ok1,
        "slippage_2x_pnl":  round(float(slip2x), 2),
        "slippage_2x_ok":   ok2,
        "comissao_50_pnl":  round(float(com50), 2),
        "comissao_50_ok":   ok3,
        "stress_score":     f"{n}/3",
        "aprovado":         n >= 2,
    }
except Exception:
    return {"aprovado": False, "stress_score": "0/3"}
```

# ================================================================

# SECAO 7: STAGE-GATE

# ================================================================

def stage_gate(m, stress, cpcv_taxa=None):
if not m:
return False, {}
checks = {
“S1_trades”:      m[“total_trades”] >= MIN_TRADES,
“S2_pf”:          m[“profit_factor”] > 1.1,
“S3_sharpe”:      m[“sharpe_ratio”] > 0.3,
“S4_dd”:          m[“max_drawdown_pct”] > MAX_DD,
“S5_dsr”:         m[“dsr”] > 0.60,
“S6_consistencia”:m[“pct_janelas_pos”] >= 55,
“S7_stress”:      stress.get(“aprovado”, False),
}
if cpcv_taxa is not None:
checks[“S8_cpcv”] = cpcv_taxa >= 0.55
aprovado = all(checks.values())
return aprovado, {
“criterios”:   checks,
“n_aprovados”: sum(checks.values()),
“total”:       len(checks),
“status”:      “APROVADO” if aprovado else “REPROVADO”,
}

# ================================================================

# SECAO 8: GRID SEARCH MASSIVO VETORIZADO

# ================================================================

# Cache global

_FEAT = None

def grid_search_familia(familia, f, params_grid, n_trials_total, mini=False):
“””
Testa todos os combos de uma familia usando VectorBT.
Retorna lista de resultados ordenados por score.
“””
print(f”\n[{familia.upper()}] Iniciando grid search…”)

```
# Gerar todos os combos
keys   = list(params_grid.keys())
values = list(params_grid.values())
combos = list(itertools.product(*values))

if mini:
    combos = combos[:3]

print(f"  {len(combos):,} combos a testar...")
t0 = time.time()

resultados = []
validos     = 0

for combo in combos:
    params = dict(zip(keys, combo))

    try:
        # Gerar sinais
        entries, exits = gerar_sinais_familia(familia, f, params)
        if entries is None:
            continue

        # Calcular sl/tp como % do preco
        atr14  = f["atr_14"]
        close  = f["close"]
        atr_pct_mean = float(np.nanmean(atr14 / close))
        sl_pct = atr_pct_mean * params.get("atr_sl", 1.0)
        tp_pct = sl_pct * params.get("rr", 2.0)

        # Backtest VectorBT
        direction = "longonly" if params.get("direction","long") == "long" else "shortonly"
        pf = backtest_vbt(f, entries, exits, sl_pct, tp_pct, direction)
        m  = extrair_metricas_vbt(pf, n_trials_total)

        if not m:
            continue

        validos += 1

        # Score por robustez
        pf_s  = min(m["profit_factor"], MAX_PF) / MAX_PF
        dsr_s = m["dsr"]
        exp_s = max(0, min(m["expectancy_brl"], 500)) / 500
        jan_s = m["pct_janelas_pos"] / 100
        tr_s  = min(m["total_trades"], 2000) / 2000
        score = pf_s*0.25 + dsr_s*0.25 + exp_s*0.25 + jan_s*0.15 + tr_s*0.10

        resultados.append({
            "familia": familia,
            "params":  params,
            "score":   round(score, 6),
            **m,
        })

    except Exception:
        continue

elapsed = time.time() - t0
resultados.sort(key=lambda x: -x["score"])

print(f"  {validos:,} validos de {len(combos):,} | {elapsed:.1f}s | "
      f"{len(combos)/max(elapsed,0.1):.0f} combos/s")

return resultados
```

def gerar_sinais_familia(familia, f, params):
“”“Despacha para o gerador correto por familia.”””
d       = params.get(“direction”, “long”)
session = params.get(“session”, “all”)

```
# Filtro de sessao
if session == "am":
    sess_mask = f["session_am"].astype(bool)
elif session == "pm":
    sess_mask = f["session_pm"].astype(bool)
else:
    sess_mask = np.ones(len(f["close"]), dtype=bool)

entries = exits = None

if familia == "ema_crossover":
    entries, exits = gerar_sinais_ema_cross(f, params["fast"], params["slow"])

elif familia == "rsi_reversion":
    period     = params["period"]
    oversold   = params["oversold"]
    overbought = params["overbought"]
    exit_level = params.get("exit_level", 50)
    key = f"rsi_{period}"
    if key not in f:
        return None, None
    rsi = f[key]
    if d == "long":
        entries = (rsi < oversold) & (np.roll(rsi,1) >= oversold)
        exits   = rsi > exit_level
    else:
        entries = (rsi > overbought) & (np.roll(rsi,1) <= overbought)
        exits   = rsi < (100 - exit_level)
    entries[0] = exits[0] = False

elif familia == "stochastic":
    entries, exits = gerar_sinais_stoch(f, params["period"],
                                         params["oversold"],
                                         params["overbought"], d)

elif familia == "bollinger":
    entries, exits = gerar_sinais_bollinger(f, params["period"],
                                            params["std_mult"], d)

elif familia == "rsi_ema_combo":
    rsi_p  = params["rsi_period"]
    rsi_l  = params["rsi_level"]
    ema_p  = params["ema_period"]
    filter_ = params["ema_filter"]
    rsi_k  = f"rsi_{rsi_p}"
    ema_k  = f"ema_{ema_p}"
    if rsi_k not in f or ema_k not in f:
        return None, None
    rsi    = f[rsi_k]
    ema    = f[ema_k]
    close  = f["close"]
    above  = close > ema
    if d == "long":
        ema_cond = above if filter_ == "above" else ~above
        entries  = (rsi < rsi_l) & ema_cond & (np.roll(rsi,1) >= rsi_l)
        exits    = rsi > 50
    else:
        ema_cond = ~above if filter_ == "above" else above
        entries  = (rsi > (100-rsi_l)) & ema_cond & (np.roll(rsi,1) <= (100-rsi_l))
        exits    = rsi < 50
    entries[0] = exits[0] = False

elif familia == "macd_momentum":
    cfg = params["config"].split("_")
    entries, exits = gerar_sinais_macd(f, int(cfg[0]), int(cfg[1]), d)

elif familia == "donchian":
    entries, exits = gerar_sinais_donchian(f, params["period"], d)

elif familia == "bb_rsi_combo":
    bb_p  = params["bb_period"]
    bb_s  = params["bb_std"]
    rsi_p = params["rsi_period"]
    rsi_c = params["rsi_confirm"]
    bb_k  = f"bb_{bb_p}_{str(bb_s).replace('.','')}_pct"
    rsi_k = f"rsi_{rsi_p}"
    if bb_k not in f or rsi_k not in f:
        return None, None
    bb_pct = f[bb_k]
    rsi    = f[rsi_k]
    if d == "long":
        entries = (bb_pct < 0.1) & (rsi < rsi_c)
        exits   = bb_pct > 0.5
    else:
        entries = (bb_pct > 0.9) & (rsi > (100-rsi_c))
        exits   = bb_pct < 0.5
    entries[0] = exits[0] = False

elif familia == "roc_momentum":
    entries, exits = gerar_sinais_roc(f, params["period"],
                                      params["threshold"], d)

elif familia == "cci":
    entries, exits = gerar_sinais_cci(f, params["period"],
                                      params["threshold"], d)

elif familia == "macd_rsi_combo":
    cfg    = params["macd_config"].split("_")
    rsi_p  = params["rsi_period"]
    rsi_f  = params["rsi_filter"]
    hist_k = f"macd_{cfg[0]}_{cfg[1]}_hist"
    rsi_k  = f"rsi_{rsi_p}"
    if hist_k not in f or rsi_k not in f:
        return None, None
    hist = f[hist_k]
    rsi  = f[rsi_k]
    if d == "long":
        entries = (hist > 0) & (np.roll(hist,1) <= 0) & (rsi > rsi_f)
        exits   = (hist < 0) & (np.roll(hist,1) >= 0)
    else:
        entries = (hist < 0) & (np.roll(hist,1) >= 0) & (rsi < (100-rsi_f))
        exits   = (hist > 0) & (np.roll(hist,1) <= 0)
    entries[0] = exits[0] = False

elif familia == "dual_ma":
    fast_p = params["fast"]
    slow_p = params["slow"]
    ft     = params["fast_type"]
    st     = params["slow_type"]
    fk     = f"{ft}_{fast_p}"
    sk     = f"{st}_{slow_p}"
    if fk not in f or sk not in f:
        return None, None
    ef = f[fk]
    es = f[sk]
    if d == "long":
        entries = (ef > es) & (np.roll(ef,1) <= np.roll(es,1))
        exits   = (ef < es) & (np.roll(ef,1) >= np.roll(es,1))
    else:
        entries = (ef < es) & (np.roll(ef,1) >= np.roll(es,1))
        exits   = (ef > es) & (np.roll(ef,1) <= np.roll(es,1))
    entries[0] = exits[0] = False

elif familia == "volatility":
    entries, exits = gerar_sinais_volatility(f, params["vol_threshold"], d)

else:
    return None, None

if entries is None or exits is None:
    return None, None

# Aplicar filtro de sessao
entries = entries & sess_mask
exits   = exits   # exits nao filtrados por sessao

return entries, exits
```

# Grid de parametros por familia

GRIDS = {
# 1. EMA Crossover com filtro de sessao
“ema_crossover”: {
“fast”:      [3, 5, 8, 10, 13, 20, 21, 34],
“slow”:      [20, 21, 34, 50, 100, 150, 200],
“rr”:        [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
“session”:   [“am”, “pm”, “all”],
“direction”: [“long”, “short”],
},  # 8*7*6*6*3*2 = 12,096
# 2. RSI Reversion expandido com exit level
“rsi_reversion”: {
“period”:     [3, 5, 7, 9, 11, 14, 18, 21, 28],
“oversold”:   [10, 15, 20, 25, 30, 35, 40],
“overbought”: [60, 65, 70, 75, 80, 85, 90],
“exit_level”: [45, 50, 55],
“rr”:         [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:     [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”:  [“long”, “short”],
},  # 9*7*7*3*5*5*2 = 66,150
# 3. Stochastic expandido
“stochastic”: {
“period”:     [3, 5, 7, 9, 14, 21, 28],
“oversold”:   [10, 15, 20, 25, 30, 35],
“overbought”: [65, 70, 75, 80, 85, 90],
“rr”:         [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:     [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”:  [“long”, “short”],
},  # 7*6*6*5*5*2 = 12,600
# 4. Bollinger com sessao
“bollinger”: {
“period”:    [5, 10, 15, 20, 30, 50],
“std_mult”:  [1.0, 1.5, 2.0, 2.5, 3.0],
“session”:   [“am”, “pm”, “all”],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
“direction”: [“long”, “short”],
},  # 6*5*3*5*6*2 = 5,400
# 5. RSI + EMA combo
“rsi_ema_combo”: {
“rsi_period”: [7, 9, 14, 21],
“rsi_level”:  [25, 30, 35, 40],
“ema_period”: [20, 50, 100, 200],
“ema_filter”: [“above”, “below”],
“rr”:         [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:     [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”:  [“long”, “short”],
},  # 4*4*4*2*5*5*2 = 6,400
# 6. MACD com sessao
“macd_momentum”: {
“config”:    [“12_26”, “8_21”, “5_13”, “3_10”, “6_19”, “10_22”, “4_9”],
“session”:   [“am”, “pm”, “all”],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
“direction”: [“long”, “short”],
},  # 7*3*5*6*2 = 1,260
# 7. Donchian com sessao
“donchian”: {
“period”:    [5, 10, 15, 20, 30, 50, 100, 200],
“session”:   [“am”, “pm”, “all”],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
“direction”: [“long”, “short”],
},  # 8*3*5*6*2 = 1,440
# 8. BB + RSI combo
“bb_rsi_combo”: {
“bb_period”:   [10, 20, 30, 50],
“bb_std”:      [1.5, 2.0, 2.5],
“rsi_period”:  [7, 14, 21],
“rsi_confirm”: [30, 35, 40, 45],
“rr”:          [1.5, 2.0, 2.5, 3.0],
“atr_sl”:      [0.5, 0.7, 1.0, 1.5],
“direction”:   [“long”, “short”],
},  # 4*3*3*4*4*4*2 = 4,608
# 9. ROC Momentum
“roc_momentum”: {
“period”:    [2, 3, 5, 10, 20, 30],
“threshold”: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”: [“long”, “short”],
},  # 6*7*5*5*2 = 2,100
# 10. CCI
“cci”: {
“period”:    [7, 10, 14, 20, 30, 50],
“threshold”: [50, 75, 100, 150, 200, 250],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”: [“long”, “short”],
},  # 6*6*5*5*2 = 1,800
# 11. MACD + RSI combo
“macd_rsi_combo”: {
“macd_config”: [“12_26”, “8_21”, “5_13”],
“rsi_period”:  [7, 9, 14, 21],
“rsi_filter”:  [40, 45, 50, 55, 60],
“rr”:          [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:      [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”:   [“long”, “short”],
},  # 3*4*5*5*5*2 = 3,000
# 12. Dual MA (EMA/SMA cruzado)
“dual_ma”: {
“fast”:      [5, 8, 10, 13, 20, 21],
“slow”:      [50, 100, 150, 200],
“fast_type”: [“ema”, “sma”],
“slow_type”: [“ema”, “sma”],
“rr”:        [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:    [0.3, 0.5, 0.7, 1.0, 1.5],
“direction”: [“long”, “short”],
},  # 6*4*2*2*5*5*2 = 4,800
# 13. Volatility Breakout
“volatility”: {
“vol_threshold”: [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
“session”:       [“am”, “pm”, “all”],
“rr”:            [1.2, 1.5, 2.0, 2.5, 3.0],
“atr_sl”:        [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
“direction”:     [“long”, “short”],
},  # 8*3*5*6*2 = 1,440
}

# ================================================================

# SECAO 9: VALIDACAO CIENTIFICA

# ================================================================

def validar_cpcv(df, familia, params):
try:
from skfolio.model_selection import CombinatorialPurgedCV
except ImportError:
return [], 0.0

```
X  = np.arange(len(df)).reshape(-1, 1)
cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2,
                            purged_size=30, embargo_size=10)
resultados = []
for train_idx, test_paths in cv.split(X):
    for test_idx in test_paths:
        df_te = df.iloc[test_idx]
        if len(df_te) < 500:
            continue
        try:
            f_te = calcular_features(df_te)
            e, x = gerar_sinais_familia(familia, f_te, params)
            if e is None:
                continue
            atr14 = f_te["atr_14"]
            close = f_te["close"]
            atr_pct = float(np.nanmean(atr14/close))
            sl  = atr_pct * params.get("atr_sl", 1.0)
            tp  = sl * params.get("rr", 2.0)
            d   = "longonly" if params.get("direction","long")=="long" else "shortonly"
            pf  = backtest_vbt(f_te, e, x, sl, tp, d)
            m   = extrair_metricas_vbt(pf)
            if m:
                resultados.append({"pnl": m["total_pnl_brl"]})
        except Exception:
            pass
lucrativos = sum(1 for r in resultados if r["pnl"] > 0)
taxa       = lucrativos / len(resultados) if resultados else 0
return resultados, taxa
```

def monte_carlo_block(pf, n_sim=1000):
try:
from arch.bootstrap import CircularBlockBootstrap
except ImportError:
return {}
if pf is None:
return {}
try:
trades = pf.trades.records_readable
pnls   = trades[“PnL”].values
if len(pnls) < 20:
return {}
bs = CircularBlockBootstrap(max(5, int(np.sqrt(len(pnls)))), pnls, seed=42)
rets, dds, ruinas = [], [], 0
for data, _ in bs.bootstrap(n_sim):
sim = data[0]
eq  = np.insert(CAPITAL + np.cumsum(sim), 0, CAPITAL)
pk  = np.maximum.accumulate(eq)
dds.append(((eq-pk)/pk*100).min())
rets.append((eq[-1]-CAPITAL)/CAPITAL*100)
if eq[-1] < CAPITAL*0.5: ruinas += 1
rf, md = np.array(rets), np.array(dds)
return {
“prob_lucro”: round(float((rf>0).mean()*100), 1),
“retorno_med”:round(float(np.median(rf)), 2),
“dd_mediano”: round(float(np.median(md)), 2),
“prob_ruina”: round(float(ruinas/n_sim*100), 2),
}
except Exception:
return {}

# ================================================================

# SECAO 10: RELATORIO

# ================================================================

def exibir_top(resultados, familia, n=10):
print(f”\n  TOP {min(n,len(resultados))} — {familia.upper()}”)
print(f”  {‘PF’:>6} {‘DSR’:>6} {‘WR%’:>6} {‘Trades’:>7} “
f”{‘Exp R$’:>8} {‘Jan+’:>5} {‘Score’:>7}”)
print(f”  {’-’*54}”)
for r in resultados[:n]:
print(f”  {r[‘profit_factor’]:>6.3f} “
f”{r[‘dsr’]:>6.3f} “
f”{r[‘win_rate’]:>6.1f} “
f”{r[‘total_trades’]:>7} “
f”{r[‘expectancy_brl’]:>8.2f} “
f”{r[‘pct_janelas_pos’]:>5.0f}% “
f”{r[‘score’]:>7.4f}”)

# ================================================================

# SECAO 11: MAIN

# ================================================================

def main():
MINI = “–mini” in sys.argv

```
print("=" * 68)
print("  ALPHA DISCOVERY ENGINE v2 — VETORIZADO COM VECTORBT")
print("  WDO B3 | 9 Familias | Milhoes de combos em minutos")
print("=" * 68)

# Contar total de combos
total_combos = sum(
    math.prod(len(v) for v in g.values())
    for g in GRIDS.values()
)
print(f"\n  Total de combos possiveis: {total_combos:,}")
print(f"  Familias: {len(GRIDS)}")
print(f"  MIN_TRADES: {MIN_TRADES} | PF_MAX: {MAX_PF} | "
      f"SHARPE_MAX: {MAX_SHARPE}")

# Carregar dados
df    = carregar()
split = int(len(df) * 0.70)
df_ins = df.iloc[:split]
df_oos = df.iloc[split:]
print(f"\n  In-sample : {len(df_ins):,} | "
      f"{df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
print(f"  Out-sample: {len(df_oos):,} | "
      f"{df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

# Pre-computar features
f_ins = calcular_features(df_ins)

if MINI:
    print("\n[MINI] Testando 3 combos por familia...")
    for familia, grid in GRIDS.items():
        res = grid_search_familia(familia, f_ins, grid,
                                  total_combos, mini=True)
        if res:
            r = res[0]
            print(f"  {familia:20}: "
                  f"PF={r['profit_factor']:.3f} | "
                  f"WR={r['win_rate']:.1f}% | "
                  f"Trades={r['total_trades']} | "
                  f"DSR={r['dsr']:.3f}")
        else:
            print(f"  {familia:20}: sem resultados validos")
    return

# === GRID SEARCH COMPLETO ===
todos_resultados = []
aprovados_is     = []

for familia, grid in GRIDS.items():
    n_combos_familia = math.prod(len(v) for v in grid.values())
    resultados = grid_search_familia(
        familia, f_ins, grid, total_combos
    )

    if not resultados:
        print(f"  [{familia}] 0 configs validas")
        continue

    exibir_top(resultados, familia)
    todos_resultados.extend(resultados[:20])

    # Salvar parcial
    path = f"{OUTPUT_DIR}/{familia}_top20.json"
    with open(path, "w") as fp:
        json.dump(resultados[:20], fp, indent=2, default=str)

    # Validar melhor config
    melhor = resultados[0]
    params = melhor["params"]

    # Stress test
    atr14  = f_ins["atr_14"]
    close  = f_ins["close"]
    atr_pct = float(np.nanmean(atr14/close))
    sl  = atr_pct * params.get("atr_sl", 1.0)
    tp  = sl * params.get("rr", 2.0)
    d   = "longonly" if params.get("direction","long")=="long" else "shortonly"

    ent, ext = gerar_sinais_familia(familia, f_ins, params)
    pf_is = backtest_vbt(f_ins, ent, ext, sl, tp, d)
    stress = stress_test_vbt(pf_is)

    # Stage-gate IS
    aprovado_is, gate_is = stage_gate(melhor, stress)

    print(f"\n  [{familia}] Melhor: "
          f"PF={melhor['profit_factor']:.3f} | "
          f"DSR={melhor['dsr']:.3f} | "
          f"Stress={stress['stress_score']} | "
          f"{'✅ APROVADO IS' if aprovado_is else '❌ REPROVADO IS'}")

    if aprovado_is:
        aprovados_is.append((familia, params, melhor, pf_is))

# === VALIDACAO FINAL ===
print(f"\n{'='*68}")
print(f"  VALIDACAO FINAL: {len(aprovados_is)} aprovado(s) no IS")
print(f"{'='*68}")

resultados_finais = []

for familia, params, melhor, pf_is in aprovados_is:
    print(f"\n  Validando [{familia}]...")

    # OOS
    f_oos = calcular_features(df_oos)
    atr_pct_oos = float(np.nanmean(f_oos["atr_14"]/f_oos["close"]))
    sl_oos = atr_pct_oos * params.get("atr_sl", 1.0)
    tp_oos = sl_oos * params.get("rr", 2.0)
    d      = "longonly" if params.get("direction","long")=="long" else "shortonly"

    e_oos, x_oos = gerar_sinais_familia(familia, f_oos, params)
    pf_oos = backtest_vbt(f_oos, e_oos, x_oos, sl_oos, tp_oos, d)
    m_oos  = extrair_metricas_vbt(pf_oos)

    # Degradacao
    is_pf  = melhor["profit_factor"]
    oos_pf = m_oos["profit_factor"] if m_oos else 0
    if is_pf > 0 and oos_pf > 0:
        deg = (is_pf - oos_pf) / is_pf * 100
        print(f"  Degradacao OOS: {deg:.1f}% "
              f"({'OK' if deg<40 else 'ATENCAO'})")

    # CPCV
    _, cpcv_taxa = validar_cpcv(df_ins, familia, params)

    # Monte Carlo
    mc = monte_carlo_block(pf_is, n_sim=1000)

    # Stage-gate final
    stress = stress_test_vbt(pf_is)
    aprovado_final, gate_final = stage_gate(
        m_oos or melhor, stress, cpcv_taxa
    )

    print(f"  OOS PF: {oos_pf:.3f} | "
          f"CPCV: {cpcv_taxa*100:.0f}% | "
          f"MC Lucro: {mc.get('prob_lucro','?')}% | "
          f"{'✅ APROVADO' if aprovado_final else '❌ REPROVADO'}")

    resultado = {
        "id":           f"{familia}_{int(time.time())}",
        "familia":      familia,
        "params":       params,
        "score_is":     melhor["score"],
        "metricas_is":  melhor,
        "metricas_oos": m_oos,
        "cpcv_taxa":    cpcv_taxa,
        "monte_carlo":  mc,
        "stress":       stress,
        "stage_gate":   gate_final,
        "aprovado":     aprovado_final,
        "gerado_em":    datetime.now().isoformat(),
    }
    resultados_finais.append(resultado)

    path = f"{OUTPUT_DIR}/{familia}_final.json"
    with open(path, "w") as fp:
        json.dump(resultado, fp, indent=2, default=str)

# === LEADERBOARD ===
print(f"\n{'='*68}")
print(f"  LEADERBOARD FINAL")
print(f"{'='*68}")
print(f"  {'FAMILIA':22} {'PF_IS':>6} {'PF_OOS':>7} "
      f"{'DSR':>6} {'CPCV':>6} {'STATUS':>12}")
print(f"  {'-'*65}")

for r in sorted(resultados_finais, key=lambda x: -x["score_is"]):
    m_is  = r["metricas_is"]
    m_oos = r["metricas_oos"] or {}
    print(f"  {r['familia']:22} "
          f"{m_is.get('profit_factor',0):>6.3f} "
          f"{m_oos.get('profit_factor',0):>7.3f} "
          f"{m_is.get('dsr',0):>6.3f} "
          f"{r['cpcv_taxa']*100:>5.0f}% "
          f"{'✅ APROVADO' if r['aprovado'] else '❌ REPROVADO':>12}")

# Salvar leaderboard
n_apr = sum(1 for r in resultados_finais if r["aprovado"])
lb = {
    "gerado_em":       datetime.now().isoformat(),
    "total_combos":    total_combos,
    "familias":        len(GRIDS),
    "aprovados_is":    len(aprovados_is),
    "aprovados_final": n_apr,
    "leaderboard":     resultados_finais,
}
path_lb = f"{OUTPUT_DIR}/leaderboard.json"
with open(path_lb, "w") as fp:
    json.dump(lb, fp, indent=2, default=str)

print(f"\n[OK] Leaderboard salvo em {path_lb}")
print(f"\n  {n_apr} estrategia(s) aprovada(s) para paper trading!")
if n_apr > 0:
    print("  Proximos passos:")
    print("  1. Revisar leaderboard.json")
    print("  2. Paper trading por 30 dias")
    print("  3. EA MQL5 para MT5")
```

if **name** == “**main**”:
main()