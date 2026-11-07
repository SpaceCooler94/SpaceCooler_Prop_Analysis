"""Prop board: recency-weighted NegBin/Poisson model + no-vig sharp blend -> edges.
Deterministic, auditable. Output: docs/data/board.json and logs/YYYY-MM-DD.json
"""
import os, json, math, time, unicodedata, requests
from datetime import datetime, timezone
import numpy as np
from scipy import stats

ODDS_KEY = os.environ["ODDS_API_KEY"]
SHARP = "pinnacle"
MIN_GAMES = 8
HALF_LIFE = 10          # games
W_RECENT = 0.6          # recent vs season mean
W_MODEL = 0.5           # model vs sharp no-vig blend
MIN_EDGE = 0.03
KELLY_FRAC, KELLY_CAP = 0.25, 0.02

MARKETS = {
    "basketball_nba": {
        "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
        "player_threes": "FG3M", "player_points_rebounds_assists": "PRA"},
    "baseball_mlb": {
        "batter_hits": "hits", "batter_home_runs": "homeRuns",
        "batter_total_bases": "totalBases", "pitcher_strikeouts": "strikeOuts"},
}

def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    for t in (".", ",", " jr", " sr", " iii", " ii"):
        n = n.replace(t, "")
    return n.strip()

# ---------- stats ----------
_cache = {}

def nba_log(name, stat):
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    key = ("nba", norm(name))
    if key not in _cache:
        m = [p for p in players.get_active_players() if norm(p["full_name"]) == norm(name)]
        df = None
        if m:
            y = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            season = f"{y}-{str(y + 1)[2:]}"
            time.sleep(0.7)
            try:
                df = playergamelog.PlayerGameLog(player_id=m[0]["id"], season=season, timeout=30).get_data_frames()[0]
            except Exception:
                df = None
        _cache[key] = df
    df = _cache[key]
    if df is None or df.empty:
        return []
    col = df["PTS"] + df["REB"] + df["AST"] if stat == "PRA" else df[stat]
    return [float(x) for x in col.tolist()]  # most recent first

def mlb_log(name, stat):
    pitching = stat == "strikeOuts"
    key = ("mlb", norm(name), pitching)
    if key not in _cache:
        out = []
        try:
            r = requests.get("https://statsapi.mlb.com/api/v1/people/search",
                             params={"names": name}, timeout=20).json()
            pid = r["people"][0]["id"]
            yr = datetime.now().year
            g = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
                             params={"stats": "gameLog", "group": "pitching" if pitching else "hitting",
                                     "season": yr}, timeout=20).json()
            out = g["stats"][0]["splits"][::-1]
        except Exception:
            pass
        _cache[key] = out
    return [float(s["stat"].get(stat, 0)) for s in _cache[key]]

def gamelog(sport, name, stat):
    return nba_log(name, stat) if sport == "basketball_nba" else mlb_log(name, stat)

# ---------- model ----------
def project(xs):
    x = np.array(xs, float)
    w = 0.5 ** (np.arange(len(x)) / HALF_LIFE)
    wm = float((w * x).sum() / w.sum())
    mu = W_RECENT * wm + (1 - W_RECENT) * float(x.mean())
    return mu, float(np.var(x, ddof=1))

def p_over(mu, var, line):
    k = math.floor(line)
    if mu <= 0:
        return 0.0
    if var > mu * 1.02:
        r = mu * mu / (var - mu)
        return float(stats.nbinom.sf(k, r, r / (r + mu)))
    return float(stats.poisson.sf(k, mu))

def novig(o, u):
    a, b = 1 / o, 1 / u
    return a / (a + b)

# ---------- odds ----------
def get(url, **p):
    p["apiKey"] = ODDS_KEY
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def run():
    rows = []
    base = "https://api.the-odds-api.com/v4/sports"
    for sport, mk in MARKETS.items():
        try:
            events = get(f"{base}/{sport}/events")
        except Exception:
            continue
        for ev in events:
            try:
                data = get(f"{base}/{sport}/events/{ev['id']}/odds", regions="us,eu",
                           markets=",".join(mk), oddsFormat="decimal")
            except Exception:
                continue
            grp = {}
            for bk in data.get("bookmakers", []):
                for m in bk["markets"]:
                    for o in m["outcomes"]:
                        k = (m["key"], o["description"], o["point"])
                        grp.setdefault(k, {}).setdefault(bk["key"], {})[o["name"]] = o["price"]
            for (market, player, line), books in grp.items():
                xs = gamelog(sport, player, mk[market])
                if len(xs) < MIN_GAMES:
                    continue
                mu, var = project(xs)
                pm = p_over(mu, var, line)
                sh = books.get(SHARP, {})
                ps = novig(sh["Over"], sh["Under"]) if "Over" in sh and "Under" in sh else None
                p_over_f = pm if ps is None else W_MODEL * pm + (1 - W_MODEL) * ps
                for book, px in books.items():
                    if book == SHARP:
                        continue
                    for side, p in (("Over", p_over_f), ("Under", 1 - p_over_f)):
                        if side not in px:
                            continue
                        dec = px[side]
                        edge = p * dec - 1
                        if edge < MIN_EDGE:
                            continue
                        pm_side = pm if side == "Over" else 1 - pm
                        ps_side = None if ps is None else (ps if side == "Over" else 1 - ps)
                        confirmed = ps_side is not None and pm_side * dec > 1 and ps_side * dec > 1
                        kelly = min(KELLY_FRAC * edge / (dec - 1), KELLY_CAP)
                        rows.append({
                            "sport": sport, "game": f"{ev['away_team']} @ {ev['home_team']}",
                            "start": ev["commence_time"], "player": player, "market": market,
                            "line": line, "side": side, "book": book, "odds": dec,
                            "proj": round(mu, 2), "n": len(xs),
                            "p_model": round(pm_side, 4),
                            "p_sharp": None if ps_side is None else round(ps_side, 4),
                            "p_fair": round(p, 4), "edge": round(edge, 4),
                            "kelly": round(kelly, 4),
                            "tier": "CONFIRMED" if confirmed else ("MODEL-ONLY" if ps is None else "UNCONFIRMED")})
    rows.sort(key=lambda r: (r["tier"] != "CONFIRMED", -r["edge"]))
    now = datetime.now(timezone.utc)
    out = {"updated": now.isoformat(), "count": len(rows), "rows": rows}
    os.makedirs("docs/data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open("docs/data/board.json", "w"))
    json.dump(out, open(f"logs/{now:%Y-%m-%d}.json", "w"))

if __name__ == "__main__":
    run()
