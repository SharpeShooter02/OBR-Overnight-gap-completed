"""
config.py — Single source of truth for all study parameters.

All pre-specified parameters are defined here and must not be modified
after the study begins. The discovery/validation split is enforced
architecturally: analysis functions accept DataFrames and the caller
selects which period to pass.
"""

import os
from pathlib import Path

# Load .env from project root if present (keeps secrets out of source control)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Instruments ───────────────────────────────────────────────────────────────
# Each entry maps a leveraged ETF to its continuously-traded underlying.
# 'inverse' : ETF moves opposite to its underlying
# 'leverage': stated leverage multiplier (informational, used for sub-analyses)

INSTRUMENTS = {
    # ── G1: Tight basket arbitrage ────────────────────────────────────────
    "TQQQ": {"underlying": "QQQ",  "inverse": False, "leverage": 3},
    "SQQQ": {"underlying": "QQQ",  "inverse": True,  "leverage": 3},
    "QLD":  {"underlying": "QQQ",  "inverse": False, "leverage": 2},
    "QID":  {"underlying": "QQQ",  "inverse": True,  "leverage": 2},
    "SPXL": {"underlying": "SPY",  "inverse": False, "leverage": 3},
    "SPXS": {"underlying": "SPY",  "inverse": True,  "leverage": 3},
    "SSO":  {"underlying": "SPY",  "inverse": False, "leverage": 2},
    "SDS":  {"underlying": "SPY",  "inverse": True,  "leverage": 2},
    "TNA":  {"underlying": "IWM",  "inverse": False, "leverage": 3},
    "TZA":  {"underlying": "IWM",  "inverse": True,  "leverage": 3},
    "FAS":  {"underlying": "XLF",  "inverse": False, "leverage": 3},
    "FAZ":  {"underlying": "XLF",  "inverse": True,  "leverage": 3},
    "FNGU": {"underlying": "QQQ",  "inverse": False, "leverage": 3},
    "FNGD": {"underlying": "QQQ",  "inverse": True,  "leverage": 3},  # Direxion Daily FANG+ Bear 3X
    "BULZ": {"underlying": "QQQ",  "inverse": False, "leverage": 3},  # MicroSectors FANG Innovation 3X Bull
    "SPXL": {"underlying": "SPY",  "inverse": False, "leverage": 3},
    "FAS":  {"underlying": "XLF",  "inverse": False, "leverage": 3},
    "BITX": {"underlying": "BTC",  "inverse": False, "leverage": 2},
    "BTFX": {"underlying": "BTC",  "inverse": False, "leverage": 2},
    "ETH":  {"underlying": "ETH",  "inverse": False, "leverage": 2},
    "ETHT": {"underlying": "ETH",  "inverse": False, "leverage": 2},
    "GDXU": {"underlying": "GDX",  "inverse": False, "leverage": 3},
    "JDST": {"underlying": "GDXJ", "inverse": True,  "leverage": 2},
    "BOIL": {"underlying": "UNG",  "inverse": False, "leverage": 2},
    "AMDG": {"underlying": "AMD",  "inverse": False, "leverage": 2},
    "AMUU": {"underlying": "AMD",  "inverse": False, "leverage": 2},
    "NVDG": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "NVDL": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "NVDW": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "NVDX": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "TSL":  {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "TSLG": {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "TSLI": {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "TSLR": {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "TSLT": {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "TSLW": {"underlying": "TSLA", "inverse": False, "leverage": 2},

    # ── G2: Commodity futures arb (partial, slower) ───────────────────────
    "BOIL": {"underlying": "UNG",  "inverse": False, "leverage": 2},
    "KOLD": {"underlying": "UNG",  "inverse": True,  "leverage": 2},
    "UCO":  {"underlying": "USO",  "inverse": False, "leverage": 2},
    "SCO":  {"underlying": "USO",  "inverse": True,  "leverage": 2},
    "OILU": {"underlying": "USO",  "inverse": False, "leverage": 2},
    "ERX":  {"underlying": "XLE",  "inverse": False, "leverage": 2},
    "ERY":  {"underlying": "XLE",  "inverse": True,  "leverage": 2},
    "UGL":  {"underlying": "GLD",  "inverse": False, "leverage": 2},
    "GLL":  {"underlying": "GLD",  "inverse": True,  "leverage": 2},
    "SHNY": {"underlying": "SLV",  "inverse": False, "leverage": 2},
    "AGQ":  {"underlying": "SLV",  "inverse": False, "leverage": 2},
    "ZSL":  {"underlying": "SLV",  "inverse": True,  "leverage": 2},
    "NUGT": {"underlying": "GDX",  "inverse": False, "leverage": 2},
    "DUST": {"underlying": "GDX",  "inverse": True,  "leverage": 2},
    "JNUG": {"underlying": "GDXJ", "inverse": False, "leverage": 2},
    "JDST": {"underlying": "GDXJ", "inverse": True,  "leverage": 2},

    # ── G3: Structural non-arb (binary event driven) ──────────────────────
    "LABU": {"underlying": "IBB",  "inverse": False, "leverage": 3},
    "LABD": {"underlying": "IBB",  "inverse": True,  "leverage": 3},
    "BIB":  {"underlying": "IBB",  "inverse": False, "leverage": 2},
    "BIS":  {"underlying": "IBB",  "inverse": True,  "leverage": 2},
    "MSOX": {"underlying": "MSOS", "inverse": False, "leverage": 2},

    # ── G4: Structural non-arb (crypto native ETF) ────────────────────────
    "ETHU": {"underlying": "ETH",  "inverse": False, "leverage": 2},
    "BITX": {"underlying": "BTC",  "inverse": False, "leverage": 2},
    "BITU": {"underlying": "BTC",  "inverse": False, "leverage": 2},
    "SBIT": {"underlying": "BTC",  "inverse": True,  "leverage": 2},  # ProShares UltraShort Bitcoin (-2x)
    # ── G4b: 1x / spot crypto ETFs (added 2026-05-12) ─────────────────────
    "BITI": {"underlying": "BTC",  "inverse": True,  "leverage": 1},  # ProShares Short Bitcoin (-1x), Oct 2021
    "IBIT": {"underlying": "BTC",  "inverse": False, "leverage": 1},  # iShares Bitcoin Trust (spot), Jan 2024
    "FBTC": {"underlying": "BTC",  "inverse": False, "leverage": 1},  # Fidelity Bitcoin (spot), Jan 2024
    "GBTC": {"underlying": "BTC",  "inverse": False, "leverage": 1},  # Grayscale Bitcoin Trust (spot), Jan 2024
    "ETHA": {"underlying": "ETH",  "inverse": False, "leverage": 1},  # iShares Ethereum Trust (spot), Jul 2024
    "ETHW": {"underlying": "ETH",  "inverse": False, "leverage": 1},  # Bitwise Ethereum ETF (spot), Jul 2024
    "CETH": {"underlying": "ETH",  "inverse": False, "leverage": 1},  # 21Shares Core ETH (spot), Jul 2024
    # ── G4c: BTC leveraged variants (non-spot) ───────────────────────────
    "BTCL": {"underlying": "BTC",  "inverse": False, "leverage": 2},  # Rex Bitcoin Strategy +2x
    "BTCZ": {"underlying": "BTC",  "inverse": True,  "leverage": 2},  # Volatility Shares -2x Bitcoin
    # ── G4d: ETH leveraged variants ──────────────────────────────────────
    "ETHD": {"underlying": "ETH",  "inverse": True,  "leverage": 2},  # Volatility Shares -2x Ether
    "ETHT": {"underlying": "ETH",  "inverse": False, "leverage": 2},  # Bitwise Ethereum Strategy +2x
    "ETU":  {"underlying": "ETH",  "inverse": False, "leverage": 2},  # ProShares Ultra Ether +2x
    # ── G4e: SOL leveraged ETFs ───────────────────────────────────────────
    "SOLT": {"underlying": "SOL",  "inverse": False, "leverage": 2},  # Volatility Shares 2x Solana
    "SOLX": {"underlying": "SOL",  "inverse": False, "leverage": 2},  # Rex-Osprey 2x Solana
    # ── G4f: XRP leveraged ETFs ───────────────────────────────────────────
    "XRPT": {"underlying": "XRP",  "inverse": False, "leverage": 2},  # Volatility Shares 2x XRP
    "UXRP": {"underlying": "XRP",  "inverse": False, "leverage": 2},  # ProShares Ultra XRP

    # ── G5: Crypto-adjacent equity ────────────────────────────────────────
    "MSTU": {"underlying": "MSTR", "inverse": False, "leverage": 2},
    "MSTX": {"underlying": "MSTR", "inverse": False, "leverage": 2},
    "CONL": {"underlying": "COIN", "inverse": False, "leverage": 2},

    # ── G6: Asian-market-driven equity sector ─────────────────────────────
    "SOXL": {"underlying": "SOXX", "inverse": False, "leverage": 3},
    "SOXS": {"underlying": "SOXX", "inverse": True,  "leverage": 3},
    "EDC":  {"underlying": "EEM",  "inverse": False, "leverage": 3},
    "EDZ":  {"underlying": "EEM",  "inverse": True,  "leverage": 3},
    "YINN": {"underlying": "FXI",  "inverse": False, "leverage": 3},
    "YANG": {"underlying": "FXI",  "inverse": True,  "leverage": 3},
    "KORU": {"underlying": "EWY",  "inverse": False, "leverage": 3},
    "TSMX": {"underlying": "TSM",  "inverse": False, "leverage": 2},

    # ── G7: Single stock leveraged (no futures) ───────────────────────────
    "NVDL": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "NVDU": {"underlying": "NVDA", "inverse": False, "leverage": 2},
    "TSLL": {"underlying": "TSLA", "inverse": False, "leverage": 2},
    "AMDL": {"underlying": "AMD",  "inverse": False, "leverage": 2},
    "AAPU": {"underlying": "AAPL", "inverse": False, "leverage": 2},
    "AMZU": {"underlying": "AMZN", "inverse": False, "leverage": 2},
    "METU": {"underlying": "META", "inverse": False, "leverage": 2},
    "MSFU": {"underlying": "MSFT", "inverse": False, "leverage": 2},
    "GGLL": {"underlying": "GOOGL","inverse": False, "leverage": 2},
    "NFXL": {"underlying": "NFLX", "inverse": False, "leverage": 2},
    "SMCL": {"underlying": "SMCI", "inverse": False, "leverage": 2},
    "GMEU": {"underlying": "GME",  "inverse": False, "leverage": 2},  # T-REX 2X Long GME Daily Target
    "APPX": {"underlying": "APP",  "inverse": False, "leverage": 2},  # Tradr 2X Long APP Daily ETF
    "SNOU": {"underlying": "SNOW", "inverse": False, "leverage": 2},  # T-REX 2X Long SNOW Daily Target
    "KMLI": {"underlying": "MELI", "inverse": False, "leverage": 2},  # KraneShares 2x MercadoLibre

    # ── G8: Structural non-arb (volatility) ───────────────────────────────
    "UVXY": {"underlying": "VIX",  "inverse": False, "leverage": 1.5},
    "UVIX": {"underlying": "VIX",  "inverse": False, "leverage": 2},
    "SVIX": {"underlying": "VIX",  "inverse": True,  "leverage": 1},
    "VIXY": {"underlying": "VIX",  "inverse": False, "leverage": 1},
    "SVXY": {"underlying": "VIX",  "inverse": True,  "leverage": 0.5},
    "VIXM": {"underlying": "VIX",  "inverse": False, "leverage": 1},

    # ── G9: Technology sector ─────────────────────────────────────────────
    "TECL": {"underlying": "XLK",  "inverse": False, "leverage": 3},  # Direxion Daily Tech Bull 3X
    "TECS": {"underlying": "XLK",  "inverse": True,  "leverage": 3},  # Direxion Daily Tech Bear 3X
    "ROM":  {"underlying": "XLK",  "inverse": False, "leverage": 2},  # ProShares Ultra Technology 2x
    "REW":  {"underlying": "XLK",  "inverse": True,  "leverage": 2},  # ProShares UltraShort Technology 2x
    "WEBL": {"underlying": "XLC",  "inverse": False, "leverage": 3},  # Direxion Daily Communication Bull 3X
    "WEBS": {"underlying": "XLC",  "inverse": True,  "leverage": 3},  # Direxion Daily Communication Bear 3X

    # ── G10: Interest rates / Treasury bonds ──────────────────────────────
    "TMF":  {"underlying": "TLT",  "inverse": False, "leverage": 3},  # Direxion Daily 20yr Treasury Bull 3X
    "TMV":  {"underlying": "TLT",  "inverse": True,  "leverage": 3},  # Direxion Daily 20yr Treasury Bear 3X
    "UBT":  {"underlying": "TLT",  "inverse": False, "leverage": 2},  # ProShares Ultra 20yr Treasury 2x
    "TBT":  {"underlying": "TLT",  "inverse": True,  "leverage": 2},  # ProShares UltraShort 20yr Treasury 2x
    "TYD":  {"underlying": "IEF",  "inverse": False, "leverage": 3},  # Direxion Daily 7-10yr Treasury Bull 3X
    "TYO":  {"underlying": "IEF",  "inverse": True,  "leverage": 3},  # Direxion Daily 7-10yr Treasury Bear 3X

    # ── G11: Healthcare ───────────────────────────────────────────────────
    "CURE": {"underlying": "XLV",  "inverse": False, "leverage": 3},  # Direxion Daily Healthcare Bull 3X

    # ── G12: Sector / Thematic ────────────────────────────────────────────
    "DPST": {"underlying": "KRE",  "inverse": False, "leverage": 3},  # Direxion Daily Regional Banks Bull 3X
    "RETL": {"underlying": "XRT",  "inverse": False, "leverage": 3},  # Direxion Daily Retail Bull 3X
    "NAIL": {"underlying": "ITB",  "inverse": False, "leverage": 3},  # Direxion Daily Homebuilders Bull 3X

    # ── G13: Additional broad market / international ──────────────────────
    "UDOW": {"underlying": "DIA",  "inverse": False, "leverage": 3},  # ProShares UltraPro Dow30 3x
    "SDOW": {"underlying": "DIA",  "inverse": True,  "leverage": 3},  # ProShares UltraPro Short Dow30 3x
    "INDL": {"underlying": "INDY", "inverse": False, "leverage": 3},  # Direxion Daily India Bull 3X
    "CWEB": {"underlying": "KWEB", "inverse": False, "leverage": 2},  # Direxion Daily China Internet Bull 2X

    # ── G14: International equity ─────────────────────────────────────────
    "MIDU": {"underlying": "MDY",  "inverse": False, "leverage": 3},  # Direxion Daily Mid Cap Bull 3X
    "EZJ":  {"underlying": "EWJ",  "inverse": False, "leverage": 2},  # ProShares Ultra MSCI Japan
    "EURL": {"underlying": "FEZ",  "inverse": False, "leverage": 3},  # Direxion Daily FTSE Europe Bull 3X
    "UPV":  {"underlying": "VGK",  "inverse": False, "leverage": 2},  # ProShares Ultra FTSE Europe
    "MEXX": {"underlying": "EWW",  "inverse": False, "leverage": 3},  # Direxion Daily MSCI Mexico Bull 3X
    "CHAU": {"underlying": "ASHR", "inverse": False, "leverage": 2},  # Direxion Daily CSI 300 China A Bull 2X
    "BRZU": {"underlying": "EWZ",  "inverse": False, "leverage": 2},  # Direxion Daily MSCI Brazil Bull 2X
    "UBR":  {"underlying": "EWZ",  "inverse": False, "leverage": 2},  # ProShares Ultra MSCI Brazil Capped

    # ── G15: Energy & Aerospace ───────────────────────────────────────────
    "DFEN": {"underlying": "ITA",  "inverse": False, "leverage": 3},  # Direxion Daily Aerospace & Defense Bull 3X
    "NRGU": {"underlying": "XOP",  "inverse": False, "leverage": 3},  # MicroSectors Big Oil Index 3X
    "GUSH": {"underlying": "XOP",  "inverse": False, "leverage": 2},  # Direxion Daily S&P Oil & Gas E&P Bull 2X

    # ── G16: Real Estate ──────────────────────────────────────────────────
    "DRN":  {"underlying": "IYR",  "inverse": False, "leverage": 3},  # Direxion Daily Real Estate Bull 3X
    "URE":  {"underlying": "IYR",  "inverse": False, "leverage": 2},  # ProShares Ultra Real Estate

    # ── G17: Sector additions ─────────────────────────────────────────────
    "BNKU": {"underlying": "KBE",  "inverse": False, "leverage": 3},  # MicroSectors U.S. Big Banks 3X
    "PILL": {"underlying": "IHE",  "inverse": False, "leverage": 3},  # Direxion Daily Pharmaceutical Bull 3X
    "WANT": {"underlying": "XLY",  "inverse": False, "leverage": 3},  # Direxion Daily Consumer Disc Bull 3X
    "UTSL": {"underlying": "XLU",  "inverse": False, "leverage": 3},  # Direxion Daily Utilities Bull 3X
    "UYM":  {"underlying": "XLB",  "inverse": False, "leverage": 2},  # ProShares Ultra Basic Materials
    "DUSL": {"underlying": "XLI",  "inverse": False, "leverage": 3},  # Direxion Daily Industrials Bull 3X
    "UXI":  {"underlying": "XLI",  "inverse": False, "leverage": 2},  # ProShares Ultra Industrials

    # ── G18: Broad market additions ───────────────────────────────────────
    "UPRO": {"underlying": "SPY",  "inverse": False, "leverage": 3},  # ProShares UltraPro S&P500
    "URTY": {"underlying": "IWM",  "inverse": False, "leverage": 3},  # ProShares UltraPro Russell 2000
    "SRTY": {"underlying": "IWM",  "inverse": True,  "leverage": 3},  # ProShares UltraPro Short Russell 2000
    "UGE":  {"underlying": "XLP",  "inverse": False, "leverage": 2},  # ProShares Ultra Consumer Goods
}

# Unique underlying symbols — all fetched via Alpha Vantage.
# "equity" → TIME_SERIES_DAILY_ADJUSTED
# "index"  → TIME_SERIES_DAILY (no adjustments; used for VIX)
# "crypto" → DIGITAL_CURRENCY_DAILY
UNDERLYINGS = {
    # Crypto
    "BTC":   "crypto",
    "ETH":   "crypto",
    # Broad market
    "SPY":   "equity",
    "QQQ":   "equity",
    "IWM":   "equity",
    # Sector equity
    "XLF":   "equity",
    "XLE":   "equity",
    "IBB":   "equity",
    "IHE":   "equity",
    "GDX":   "equity",
    "GDXJ":  "equity",
    "SOXX":  "equity",
    "SMH":   "equity",
    "EEM":   "equity",
    "VWO":   "equity",
    "FXI":   "equity",
    "EWY":   "equity",
    # Commodities (ETF proxies)
    "GLD":   "equity",
    "SLV":   "equity",
    "USO":   "equity",
    "UNG":   "equity",
    # Single stock
    "COIN":  "equity",
    "MSTR":  "equity",
    "NVDA":  "equity",
    "TSLA":  "equity",
    "AAPL":  "equity",
    "AMZN":  "equity",
    "META":  "equity",
    "MSFT":  "equity",
    "GOOGL": "equity",
    "NFLX":  "equity",
    "TSM":   "equity",
    "SMCI":  "equity",   # Super Micro Computer (SMCL)
    "GME":   "equity",   # GameStop (GMEU)
    "APP":   "equity",   # AppLovin (APPX)
    "SNOW":  "equity",   # Snowflake (SNOU)
    "MELI":  "equity",   # MercadoLibre (KMLI)
    # Crypto underlyings for new leveraged ETFs
    "SOL":   "crypto",
    "XRP":   "crypto",
    # Equity underlyings for new sector groups
    "XLK":   "equity",   # Technology Select Sector
    "XLC":   "equity",   # Communication Services Select Sector
    "XLV":   "equity",   # Health Care Select Sector
    "IYR":   "equity",   # iShares US Real Estate
    "TLT":   "equity",   # iShares 20+ Year Treasury Bond
    "IEF":   "equity",   # iShares 7-10 Year Treasury Bond
    "KRE":   "equity",   # SPDR S&P Regional Banking
    "XRT":   "equity",   # SPDR S&P Retail
    "ITB":   "equity",   # iShares US Home Construction
    "DIA":   "equity",   # SPDR Dow Jones Industrial Average
    "INDY":  "equity",   # iShares India 50
    "KWEB":  "equity",   # KraneShares CSI China Internet
    # Underlyings for EVALUATE/CONFIRM candidates (added 2026-05-16)
    "ASHR":  "equity",   # Xtrackers Harvest CSI 300 China A-Shares (CHAU)
    "ITA":   "equity",   # iShares U.S. Aerospace & Defense (DFEN)
    "XOP":   "equity",   # SPDR S&P Oil & Gas E&P (NRGU, GUSH)
    "EWJ":   "equity",   # iShares MSCI Japan (EZJ)
    "MDY":   "equity",   # SPDR S&P MidCap 400 (MIDU)
    "KBE":   "equity",   # SPDR S&P Bank (BNKU)
    "EWW":   "equity",   # iShares MSCI Mexico (MEXX)
    "FEZ":   "equity",   # SPDR EURO STOXX 50 (EURL)
    "VGK":   "equity",   # Vanguard FTSE Europe (UPV)
    "EWZ":   "equity",   # iShares MSCI Brazil (BRZU, UBR)
    "XLY":   "equity",   # Consumer Discretionary Select Sector SPDR (WANT)
    "XLU":   "equity",   # Utilities Select Sector SPDR (UTSL)
    "XLB":   "equity",   # Materials Select Sector SPDR (UYM)
    "XLP":   "equity",   # Consumer Staples Select Sector SPDR (UGE)
    "XLI":   "equity",   # Industrials Select Sector SPDR (DUSL/UXI; NBI→IBB proxy already in config)
    # Crypto underlyings for new leveraged ETFs (added 2026-05-16)
    "LINK":  "crypto",   # Chainlink (CHNU)
    "ADA":   "crypto",   # Cardano (CRDX)
    "XLM":   "crypto",   # Stellar (STLU)
    # Single-stock underlyings (added 2026-05-16)
    "AMD":   "equity",   # Advanced Micro Devices (AMDL, AMDW)
}

# ── Alpha Vantage API ─────────────────────────────────────────────────────────
# Used for all data: ETF intraday (TIME_SERIES_INTRADAY) and underlying daily.
AV_API_KEY           = os.environ.get("AV_API_KEY", "")
AV_BASE_URL          = "https://www.alphavantage.co/query"
AV_CALLS_PER_MIN     = 60           # premium $50/month plan
AV_INTRADAY_INTERVAL = "1min"

# ── Study Parameters ──────────────────────────────────────────────────────────
# PRE-SPECIFIED before any outcome data was examined.
# These values must not be changed after the study begins.

# Discovery period: all hypothesis development and analysis
DISCOVERY_START    = "2008-01-01"
DISCOVERY_END      = "2022-12-31"

# Validation period: held out, touched only for final out-of-sample check
VALIDATION_START   = "2023-01-01"
VALIDATION_END     = "2026-12-31"

# Minimum absolute overnight gap to qualify as an event (%).
# Below 2% the gap represents noise rather than genuine overnight information.
MIN_GAP_PCT        = 2.0

# Fixed time points after market open at which returns are measured (minutes).
# All intervals are reported; none are selected based on outcomes.
MEASUREMENT_INTERVALS = [15, 30, 45, 60, 90, 120, 180, 240, 390]

TIMEZONE           = "America/New_York"

# ── Data Cache ────────────────────────────────────────────────────────────────
CACHE_DIR          = "cache/"
INTRADAY_DIR       = "cache/intraday/"     # 1-min bars: symbol/YYYY-MM.parquet
DAILY_DIR          = "cache/daily/"        # daily OHLCV: symbol.parquet
INDICATORS_DIR     = "cache/indicators/"   # pre-computed indicators: symbol/INDICATOR_params.parquet

# Legacy intraday cache (pre-existing data, see data/legacy_loader.py)
LEGACY_CACHE_DIR   = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'BacktestingGaps/orb_event_study/cache/intraday')

# ── Instrument categories for sub-group H1 analysis ──────────────────────────
# Grouped by gap-absorption mechanism. Pre-specified before examining outcomes.
CATEGORIES = {
    "G1_basket_arb": {
        "label":   "Group 1 — Tight basket arbitrage",
        "symbols": ["TQQQ", "SQQQ", "QLD", "QID", "SPXL", "SPXS", "SSO", "SDS",
                    "TNA", "TZA", "FAS", "FAZ", "FNGU", "FNGD"],
    },
    "G2_commodity_arb": {
        "label":   "Group 2 — Commodity futures arb (partial, slower)",
        "symbols": ["BOIL", "KOLD", "UCO", "SCO", "OILU", "ERX",
                    "UGL", "GLL", "SHNY", "AGQ", "ZSL",
                    "NUGT", "DUST", "JNUG", "JDST"],
    },
    "G3_binary_event": {
        "label":   "Group 3 — Structural non-arb (binary event driven)",
        "symbols": ["LABU", "LABD", "BIB", "BIS"],
    },
    "G4_crypto_native": {
        "label":   "Group 4 — Structural non-arb (crypto native ETF)",
        "symbols": ["ETHU", "ETHD", "ETHT", "ETU", "CETH",
                    "BITX", "BITU", "BTCL", "BTCZ", "SBIT", "BITI",
                    "IBIT", "FBTC", "GBTC", "ETHA", "ETHW",
                    "SOLT", "SOLX", "XRPT", "UXRP"],
    },
    "G5_crypto_equity": {
        "label":   "Group 5 — Crypto-adjacent equity",
        "symbols": ["MSTU", "MSTX", "CONL"],
    },
    "G6_asian_sector": {
        "label":   "Group 6 — Asian-market-driven equity sector",
        "symbols": ["SOXL", "SOXS", "EDC", "EDZ", "YINN", "YANG", "KORU", "TSMX"],
    },
    "G7_single_stock": {
        "label":   "Group 7 — Single stock leveraged (no futures)",
        "symbols": ["NVDL", "NVDU", "TSLL", "AAPU", "AMZU", "METU",
                    "MSFU", "GGLL", "NFXL", "SMCL",
                    "GMEU", "APPX", "SNOU", "KMLI"],
    },
    "G8_volatility": {
        "label":   "Group 8 — Structural non-arb (volatility)",
        "symbols": ["UVXY", "UVIX", "SVIX", "VIXY", "SVXY", "VIXM"],
    },
    "G9_technology": {
        "label":   "Group 9 — Technology sector",
        "symbols": ["TECL", "TECS", "ROM", "REW", "WEBL", "WEBS"],
    },
    "G10_rates": {
        "label":   "Group 10 — Interest rates / Treasury bonds",
        "symbols": ["TMF", "TMV", "UBT", "TBT", "TYD", "TYO"],
    },
    "G11_healthcare": {
        "label":   "Group 11 — Healthcare",
        "symbols": ["CURE"],
    },
    "G12_sector_thematic": {
        "label":   "Group 12 — Sector / Thematic",
        "symbols": ["DPST", "RETL", "NAIL"],
    },
    "G13_intl_broad": {
        "label":   "Group 13 — Additional broad market / international",
        "symbols": ["UDOW", "SDOW", "INDL", "CWEB"],
    },
}
