"""Prop board: NFL / NBA / NHL. Recency-weighted NegBin/Poisson model
+ no-vig sharp blend -> edges. Output: docs/data/board.json, logs/YYYY-MM-DD.json
"""
import os, json, math, time, unicodedata, requests
from datetime import datetime, timezone
import numpy as np
from scipy import stats

ODDS_KEY = os.environ["ODDS_API_KEY"]
SHARP = "pinnacle"
HALF_LIFE = 10
W_RECENT = 0.6
W_MODEL = 0.5
MIN_EDGE = 0.03
KELLY_FRAC, KELLY_CAP = 0.25, 0.02
MIN_GAMES = {"americanfootball_nfl": 5, "basketball_nba": 8, "icehockey_nhl": 8}

MARKETS = {
    "americanfootball_nfl": {
        "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards", "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards"},
    "basketball_nba": {
        "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
        "player_threes": "FG3M", "player_points_rebounds_assists": "PRA"},
    "icehockey_nhl": {
        "player_points": "points", "player_assists": "assists",
        "player_shots_on_goal": "shots", "player_goals": "goals"},
}

def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    for t in (".", ",", "'", " jr", " sr", " iii", " ii"):
        n = n.replace(t, "")
    return n.strip()

# ---------- stats (all return most-recent-first lists) ----------
_cache = {}

def nfl_log(name, stat):
    if "nfl" not in _cache:
        try:
            import nflreadpy as nfl
            now = datetime.now()
            yr = now.year if now.month >= 8 else now.year - 1
            df = nfl.load_player_stats([yr]).to_pandas()
            if "season_type" in df.columns:
                df = df[df["season_type"] == "REG"]
            df["_n"] = df["player_display_name"].map(norm)
            _cache["nfl"] = df.sort_values("week", ascending=False)
        except Exception as e:
            print("nfl load ERROR:", e)
            _cache["nfl"] = None
    df = _cache["nfl"]
    if df is None:
        return []
    s = df[df["_n"] == norm(name)][stat].dropna()
    return [float(x) for x in s.tolist()]

def nba_log(name, stat):
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    key = ("nba", norm(name))
    if key not in _cache:
        m = [p for p in players.get_active_players() if norm(p["full_name"]) == norm(name)]
        df = None
        if m:
            now = datetime.now()
            y = now.year if now.month >= 10 else now.year - 1
            time.sleep(0.7)
            try:
                df = playergamelog.PlayerGameLog(
                    player_id=m[0]["id"], season=f"{y}-{str(y + 1)[2:]}", timeout=30
                ).get_data_frames()[0]
            except Exception as e:
                print("nba_api ERROR:", name, e)
        _cache[key] = df
    df = _cache[key]
    if df is None or df.empty:
        return []
    col = df["PTS"] + df["REB"] + df["AST"] if stat == "PRA" else df[stat]
    return [float(x) for x in col.tolist()]

def nhl_log(name, stat):
    key = ("nhl", norm(name))
    if key not in _cache:
        out = []
        try:
            r = requests.get("https://search.d3.nhle.com/api/v1/search/player",
                             params={"culture": "en-us", "limit": 5, "q": name, "active": "true"},
                             timeout=20).json()
            hit = next((p for p in r if norm(p["name"]) == norm(name)), r[0] if r else None)
            if hit:
                g = requests.get(f"https://api-web.nhle.com/v1/player/{hit['playerId']}/game-log/now",
                                 timeout=20).json()
                out = g.get("gameLog", [])
        except Exception as e:
            print("nhl ERROR:", name, e)
        _cache[key] = out
    return [float(x.get(stat, 0)) for x in _cache[key]]

def gamelog(sport, name, stat):
    if sport == "americanfootball_nfl":
        return nfl_log(name, stat)
    if sport == "basketball_nba":
        return nba_log(name, stat)
    return nhl_log(name, stat)

# ---------- model ----------
def project(xs):
    x = np.array(xs, float)
    w = 0.5 ** (np.arange(len(x)) / HALF_LIFE)
    wm = float((w * x).sum() / w.sum())
    return W_RECENT * wm + (1 - W_RECENT) * float(x.mean()), float(np.var(x, ddof=1))

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
            print(sport, "events:", len(events))
        except Exception as e:
            print(sport, "events ERROR:", e)
            continue
        for ev in events:
            try:
                data = get(f"{base}/{sport}/events/{ev['id']}/odds", regions="us,eu",
                           markets=",".join(mk), oddsFormat="decimal")
            except Exception as e:
                resp = getattr(e, "response", None)
                print("odds ERROR:", ev["id"], resp.text[:200] if resp is not None else e)
                continue
            grp = {}
            for bk in data.get("bookmakers", []):
                for m in bk["markets"]:
                    for o in m["outcomes"]:
                        k = (m["key"], o["description"], o["point"])
                        grp.setdefault(k, {}).setdefault(bk["key"], {})[o["name"]] = o["price"]
            for (market, player, line), books in grp.items():
                xs = gamelog(sport, player, mk[market])
                if len(xs) < MIN_GAMES[sport]:
                    print("skip no log:", sport, player, mk[market], len(xs))
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
                        pm_s = pm if side == "Over" else 1 - pm
                        ps_s = None if ps is None else (ps if side == "Over" else 1 - ps)
                        confirmed = ps_s is not None and pm_s * dec > 1 and ps_s * dec > 1
                        kelly = min(KELLY_FRAC * edge / (dec - 1), KELLY_CAP)
                        rows.append({
                            "sport": sport, "game": f"{ev['away_team']} @ {ev['home_team']}",
                            "start": ev["commence_time"], "player": player, "market": market,
                            "line": line, "side": side, "book": book, "odds": dec,
                            "proj": round(mu, 2), "n": len(xs),
                            "p_model": round(pm_s, 4),
                            "p_sharp": None if ps_s is None else round(ps_s, 4),
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
    print("rows:", len(rows))

if __name__ == "__main__":
    run()
