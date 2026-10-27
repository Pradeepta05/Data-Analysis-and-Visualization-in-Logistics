#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output3")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
R = {}

# ----------------------------------------------------------------------------
# colours: one fixed order for categories, one blue ramp for magnitude
# ----------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e3e2de"
VEH_COL = {"Van": BLUE, "Mini-truck": ORANGE, "Truck": AQUA}
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
SEQ = sns.light_palette(BLUE, as_cmap=True)

# ============================================================================
# PART A - SIMULATION
# ============================================================================
N = 12000
days = pd.date_range("2025-01-01", "2025-12-31")
# more shipments Mon-Sat, festive bump in Oct-Nov, monsoon dip a little
w = np.array([1.15 if d.dayofweek == 0 else 0.25 if d.dayofweek == 6 else 1.0 for d in days])
w *= np.array([1.25 if d.month in (10, 11) else 1.0 for d in days])
date = rng.choice(days, N, p=w / w.sum())
df = pd.DataFrame({"date": pd.to_datetime(date)})
df["month"] = df.date.dt.month
df["weekday"] = df.date.dt.dayofweek

hours = np.arange(6, 21)
hp = np.array([2, 5, 14, 16, 12, 6, 4, 4, 7, 8, 6, 4, 3, 2, 1], dtype=float)
df["dispatch_hour"] = rng.choice(hours, N, p=hp / hp.sum())

regions = ["Central", "East", "North", "South", "West"]
df["region"] = rng.choice(regions, N, p=[0.30, 0.22, 0.18, 0.17, 0.13])
dist_mu = {"Central": 14, "East": 30, "North": 46, "South": 62, "West": 85}
df["distance_km"] = [max(4, rng.normal(dist_mu[r], dist_mu[r] * 0.22)) for r in df.region]
df["distance_km"] = df.distance_km.round(1)

df["volume_units"] = np.clip(rng.lognormal(np.log(260), 0.75, N), 20, 1800).round().astype(int)
veh = np.where(df.volume_units <= 300, "Van", np.where(df.volume_units <= 800, "Mini-truck", "Truck"))
mis = rng.random(N) < 0.10                           # sometimes a bigger vehicle than needed is sent
veh = np.where(mis & (veh == "Van"), "Mini-truck", np.where(mis & (veh == "Mini-truck"), "Truck", veh))
df["vehicle"] = veh
df["driver_exp_yrs"] = rng.integers(0, 16, N)

month_rain = {m: (0.35 if m in (6, 7, 8, 9) else 0.08) for m in range(1, 13)}
wx = []
for m in df.month:
    r = rng.random()
    if r < month_rain[m]: wx.append("Rain")
    elif m in (12, 1) and r < month_rain[m] + 0.12: wx.append("Fog")
    else: wx.append("Clear")
df["weather"] = wx

# loading wait at the dock depends on how many trucks leave in the same hour
slot = df.groupby(["date", "dispatch_hour"]).date.transform("size")
df["slot_load"] = slot
df["loading_wait_min"] = (5 + 3.0 * (slot - 1) + rng.exponential(6, N)).round(1)

speed = df.vehicle.map({"Van": 28, "Mini-truck": 26, "Truck": 24}).astype(float)
speed *= df.weather.map({"Clear": 1.0, "Rain": 0.78, "Fog": 0.85})
speed *= np.where(df.dispatch_hour.isin([8, 9, 17, 18]), 0.88, 1.0)
speed *= 1 + 0.008 * df.driver_exp_yrs
df["transit_hrs"] = (df.distance_km / speed * rng.lognormal(0, 0.12, N)).round(2)
df["unload_hrs"] = (0.25 + df.volume_units / 4000 + rng.exponential(0.08, N)).round(2)
df["delivery_hrs"] = (df.loading_wait_min / 60 + df.transit_hrs + df.unload_hrs).round(2)
df["scheduled_hrs"] = (df.distance_km / 30 + 1.0).round(2)
df["is_late"] = (df.delivery_hrs > df.scheduled_hrs).astype(int)

rate_km = df.vehicle.map({"Van": 14, "Mini-truck": 20, "Truck": 32})
handle = df.vehicle.map({"Van": 1.2, "Mini-truck": 0.9, "Truck": 0.7})
cost = 250 + df.distance_km * rate_km + df.volume_units * handle + df.loading_wait_min * 2.0
cost *= np.where(df.weather == "Rain", 1.08, 1.0)
cost *= 1 - 0.01 * np.minimum(df.driver_exp_yrs, 10)
cost *= rng.normal(1, 0.05, N)
df["transport_cost"] = cost.round(0)
df["cost_per_unit"] = (df.transport_cost / df.volume_units).round(2)
dmg = rng.random(N) < (0.02 + np.where(df.weather == "Rain", 0.015, 0))
df["damaged"] = dmg.astype(int)
df["rating"] = np.clip(np.round(4.6 - 1.3 * df.is_late - 1.4 * df.damaged + rng.normal(0, 0.5, N)), 1, 5).astype(int)
df.insert(0, "shipment_id", np.arange(1, N + 1))
df.to_csv(f"{OUT}/shipments_week3.csv", index=False)

R["dataset"] = {"rows": len(df), "cols": df.shape[1], "missing_cells": int(df.isna().sum().sum()),
                "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date())}

# ============================================================================
# PART B - EDA
# ============================================================================
key = ["volume_units", "distance_km", "loading_wait_min", "transit_hrs", "delivery_hrs",
       "transport_cost", "cost_per_unit", "rating"]
desc = df[key].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
R["describe"] = desc.to_dict("index")
R["mode_vehicle"] = df.vehicle.mode()[0]
R["share_vehicle"] = (df.vehicle.value_counts(normalize=True) * 100).round(1).to_dict()
R["share_region"] = (df.region.value_counts(normalize=True) * 100).round(1).to_dict()
R["otd_pct"] = round(100 - df.is_late.mean() * 100, 1)
R["late_pct"] = round(df.is_late.mean() * 100, 1)
R["damage_pct"] = round(df.damaged.mean() * 100, 2)
R["avg_rating"] = round(df.rating.mean(), 2)
R["rating_late_vs_ontime"] = df.groupby("is_late").rating.mean().round(2).to_dict()

corr_cols = ["volume_units", "distance_km", "loading_wait_min", "transit_hrs", "delivery_hrs",
             "transport_cost", "cost_per_unit", "driver_exp_yrs", "is_late"]
corr = df[corr_cols].corr().round(2)
R["corr"] = corr.to_dict()

# group tables
reg = df.groupby("region").agg(shipments=("shipment_id", "count"), avg_dist=("distance_km", "mean"),
                               avg_hrs=("delivery_hrs", "mean"), late_pct=("is_late", lambda s: s.mean() * 100),
                               cpu=("cost_per_unit", "mean"), avg_cost=("transport_cost", "mean")).round(2)
R["by_region"] = reg.to_dict("index")
veh_t = df.groupby("vehicle").agg(shipments=("shipment_id", "count"), avg_vol=("volume_units", "mean"),
                                  avg_cost=("transport_cost", "mean"), cpu=("cost_per_unit", "mean"),
                                  late_pct=("is_late", lambda s: s.mean() * 100)).round(2)
R["by_vehicle"] = veh_t.to_dict("index")
wx_t = df.groupby("weather").agg(shipments=("shipment_id", "count"), avg_hrs=("delivery_hrs", "mean"),
                                 late_pct=("is_late", lambda s: s.mean() * 100), cpu=("cost_per_unit", "mean")).round(2)
R["by_weather"] = wx_t.to_dict("index")
mon = df.groupby("month").agg(shipments=("shipment_id", "count"), volume=("volume_units", "sum"),
                              late_pct=("is_late", lambda s: s.mean() * 100), cpu=("cost_per_unit", "mean")).round(2)
R["by_month"] = mon.to_dict("index")
hr = df.groupby("dispatch_hour").agg(shipments=("shipment_id", "count"), wait=("loading_wait_min", "mean"),
                                     late_pct=("is_late", lambda s: s.mean() * 100)).round(2)
R["by_hour"] = hr.to_dict("index")
wd = df.groupby("weekday").agg(shipments=("shipment_id", "count"), wait=("loading_wait_min", "mean"),
                               late_pct=("is_late", lambda s: s.mean() * 100)).round(2)
R["by_weekday"] = wd.to_dict("index")
df["exp_band"] = pd.cut(df.driver_exp_yrs, [-1, 2, 5, 9, 15], labels=["0-2", "3-5", "6-9", "10-15"])
ex = df.groupby("exp_band", observed=True).agg(late_pct=("is_late", lambda s: s.mean() * 100),
                                               cpu=("cost_per_unit", "mean"), hrs=("delivery_hrs", "mean")).round(2)
R["by_experience"] = ex.to_dict("index")
R["outliers_iqr_delivery_hrs"] = int(((df.delivery_hrs > df.delivery_hrs.quantile(.75) + 1.5 * (df.delivery_hrs.quantile(.75) - df.delivery_hrs.quantile(.25)))).sum())

# peak vs off-peak
peak = df.dispatch_hour.isin([8, 9, 10])
R["peak"] = {"share_shipments_pct": round(peak.mean() * 100, 1),
             "wait_peak": round(df.loc[peak, "loading_wait_min"].mean(), 1),
             "wait_offpeak": round(df.loc[~peak, "loading_wait_min"].mean(), 1),
             "late_peak": round(df.loc[peak, "is_late"].mean() * 100, 1),
             "late_offpeak": round(df.loc[~peak, "is_late"].mean() * 100, 1)}

# time breakdown by region (hours)
tb = df.assign(wait_hrs=df.loading_wait_min / 60).groupby("region")[["wait_hrs", "transit_hrs", "unload_hrs"]].mean().round(2)
R["time_breakdown"] = tb.to_dict("index")
R["wait_share_of_total_pct"] = round(df.loading_wait_min.sum() / 60 / df.delivery_hrs.sum() * 100, 1)

# ============================================================================
# PART C - VISUALIZATIONS
# ============================================================================
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# V1 distributions: three histograms with mean/median lines
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, col, title, xl in [(axes[0], "delivery_hrs", "Delivery time", "hours"),
                           (axes[1], "cost_per_unit", "Transport cost per unit", "Rs per unit"),
                           (axes[2], "volume_units", "Shipment volume", "units")]:
    ax.hist(df[col], bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(df[col].mean(), color=ORANGE, lw=2, label=f"mean {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color=INK, lw=1.5, ls="--", label=f"median {df[col].median():.1f}")
    ax.set_title(title); ax.set_xlabel(xl); ax.set_yticks([]); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "v1_distributions")

# V2 boxplots by vehicle
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
order = ["Van", "Mini-truck", "Truck"]
sns.boxplot(data=df, x="vehicle", y="cost_per_unit", order=order, palette=VEH_COL, hue="vehicle", legend=False,
            ax=axes[0], linewidth=1, fliersize=2)
axes[0].set_title("Cost per unit by vehicle"); axes[0].set_ylabel("Rs per unit"); axes[0].set_xlabel("")
sns.boxplot(data=df, x="vehicle", y="volume_units", order=order, palette=VEH_COL, hue="vehicle", legend=False,
            ax=axes[1], linewidth=1, fliersize=2)
axes[1].set_title("Shipment volume by vehicle"); axes[1].set_ylabel("units"); axes[1].set_xlabel("")
fig.tight_layout(); save(fig, "v2_vehicle_box")

# V3 monthly trend: volume bars, then late% line as separate chart (no dual axis)
fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
mlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
axes[0].bar(mlabels, mon.volume / 1000, color=BLUE, width=0.6)
axes[0].set_title("Units shipped per month (thousands)"); axes[0].set_ylabel("thousand units")
axes[1].plot(mlabels, mon.late_pct, color=ORANGE, lw=2, marker="o", ms=5)
axes[1].set_title("Late deliveries per month (%)"); axes[1].set_ylabel("% late")
axes[1].set_ylim(0, max(mon.late_pct) * 1.25)
fig.tight_layout(); save(fig, "v3_monthly")

# V4 scatter distance vs cost
fig, ax = plt.subplots(figsize=(7, 4.2))
for v in order:
    s = df[df.vehicle == v].sample(min(1500, (df.vehicle == v).sum()), random_state=1)
    ax.scatter(s.distance_km, s.transport_cost, s=9, alpha=0.45, color=VEH_COL[v], label=v, linewidths=0)
    b = np.polyfit(df[df.vehicle == v].distance_km, df[df.vehicle == v].transport_cost, 1)
    xs = np.linspace(df.distance_km.min(), df.distance_km.max(), 50)
    ax.plot(xs, b[0] * xs + b[1], color=VEH_COL[v], lw=2)
ax.set_xlabel("distance (km)"); ax.set_ylabel("transport cost (Rs)")
ax.set_title("Transport cost rises with distance, at a different rate for each vehicle")
ax.legend(frameon=False, markerscale=2)
fig.tight_layout(); save(fig, "v4_distance_cost")

# V5 correlation heatmap (diverging blue-grey-orange)
div = sns.blend_palette([ORANGE, "#f2f1ee", BLUE], as_cmap=True)
fig, ax = plt.subplots(figsize=(7.4, 5.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=div, vmin=-1, vmax=1, center=0, linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.7}, annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation between key variables"); plt.xticks(rotation=40, ha="right")
fig.tight_layout(); save(fig, "v5_corr")

# V6 heatmap: wait by hour x weekday
piv = df.pivot_table(index="weekday", columns="dispatch_hour", values="loading_wait_min", aggfunc="mean")
piv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(8.4, 3.8))
sns.heatmap(piv, cmap=SEQ, annot=True, fmt=".0f", annot_kws={"size": 7}, linewidths=1, linecolor="white",
            cbar_kws={"label": "average wait (min)", "shrink": 0.8}, ax=ax)
ax.set_xlabel("dispatch hour"); ax.set_ylabel(""); ax.set_title("Average dock waiting time (minutes)")
fig.tight_layout(); save(fig, "v6_wait_heatmap")

# V7 late% by dispatch hour + shipments per hour (two charts)
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].bar(hr.index, hr.shipments, color=BLUE, width=0.65)
axes[0].set_title("Shipments dispatched per hour of day"); axes[0].set_ylabel("shipments")
axes[1].bar(hr.index, hr.late_pct, color=ORANGE, width=0.65)
axes[1].set_title("Late deliveries by dispatch hour (%)"); axes[1].set_ylabel("% late"); axes[1].set_xlabel("dispatch hour")
axes[1].set_xticks(hr.index)
fig.tight_layout(); save(fig, "v7_hour")

# V8 region comparison
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
rs = reg.reindex(regions)
axes[0].bar(rs.index, rs.late_pct, color=BLUE, width=0.6); axes[0].set_title("Late deliveries by region (%)"); axes[0].set_ylabel("% late")
axes[1].bar(rs.index, rs.cpu, color=BLUE, width=0.6); axes[1].set_title("Cost per unit by region (Rs)"); axes[1].set_ylabel("Rs per unit")
for ax, vals in zip(axes, [rs.late_pct, rs.cpu]):
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
fig.tight_layout(); save(fig, "v8_region")

# V9 weather effect
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
wsel = wx_t.reindex(["Clear", "Rain", "Fog"])
axes[0].bar(wsel.index, wsel.avg_hrs, color=BLUE, width=0.55); axes[0].set_title("Average delivery time (hrs)")
axes[1].bar(wsel.index, wsel.late_pct, color=BLUE, width=0.55); axes[1].set_title("Late deliveries (%)")
for ax, vals in zip(axes, [wsel.avg_hrs, wsel.late_pct]):
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
fig.tight_layout(); save(fig, "v9_weather")

# V10 stacked bar: where does the delivery time go?
fig, ax = plt.subplots(figsize=(7.4, 3.6))
tbr = tb.reindex(regions)
bottom = np.zeros(len(tbr))
for col, name, colr in [("wait_hrs", "Dock waiting", ORANGE), ("transit_hrs", "Driving", BLUE), ("unload_hrs", "Unloading", AQUA)]:
    ax.bar(tbr.index, tbr[col], bottom=bottom, label=name, color=colr, width=0.6, edgecolor="white", linewidth=1.5)
    bottom += tbr[col].values
ax.set_ylabel("average hours"); ax.set_title("Where delivery time is spent, by region")
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout(); save(fig, "v10_time_breakdown")

# V11 driver experience
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.plot(ex.index.astype(str), ex.late_pct, color=BLUE, lw=2, marker="o", ms=6)
for i, v in enumerate(ex.late_pct): ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xlabel("driver experience (years)"); ax.set_ylabel("% late"); ax.set_title("Late deliveries by driver experience")
ax.set_ylim(0, ex.late_pct.max() * 1.3)
fig.tight_layout(); save(fig, "v11_experience")

# V12 cost per unit vs volume (economies of scale), median per volume bin
df["vol_bin"] = pd.qcut(df.volume_units, 10)
vb = df.groupby("vol_bin", observed=True).agg(v=("volume_units", "median"), c=("cost_per_unit", "median"))
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.plot(vb.v, vb.c, color=ORANGE, lw=2, marker="o", ms=5)
ax.set_xlabel("shipment volume (units, bin median)"); ax.set_ylabel("median Rs per unit")
ax.set_title("Bigger shipments cost less per unit")
fig.tight_layout(); save(fig, "v12_scale")
R["scale_cpu_first_last"] = [round(float(vb.c.iloc[0]), 2), round(float(vb.c.iloc[-1]), 2)]

# ============================================================================
# PART D - COST DRIVERS AND WHAT-IF
# ============================================================================
X = pd.get_dummies(df[["distance_km", "volume_units", "loading_wait_min", "driver_exp_yrs", "vehicle", "weather"]],
                   columns=["vehicle", "weather"], drop_first=True, dtype=int)
Xs = StandardScaler().fit_transform(X)
lr = LinearRegression().fit(Xs, df.transport_cost)
coef = pd.Series(lr.coef_, index=X.columns).sort_values()
R["cost_model"] = {"r2": round(lr.score(Xs, df.transport_cost), 3), "std_coefs": coef.round(1).to_dict()}
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.barh(coef.index, coef.values, color=[ORANGE if v < 0 else BLUE for v in coef.values], height=0.6)
ax.set_xlabel("effect on cost (Rs) of a 1 standard-deviation increase"); ax.set_title("What drives transport cost?")
fig.tight_layout(); save(fig, "v13_cost_drivers")

# what-if: peak-hour shipments wait only as long as off-peak ones
offpeak_med = df.loc[~peak, "loading_wait_min"].median()
d2 = df.copy()
d2.loc[peak, "loading_wait_min"] = np.minimum(d2.loc[peak, "loading_wait_min"], offpeak_med + 4)
d2["delivery_hrs"] = d2.loading_wait_min / 60 + d2.transit_hrs + d2.unload_hrs
late2 = (d2.delivery_hrs > d2.scheduled_hrs).mean() * 100
cost2 = d2.transport_cost - (df.loading_wait_min - d2.loading_wait_min) * 2.0
R["whatif"] = {"late_before": round(df.is_late.mean() * 100, 1), "late_after": round(late2, 1),
               "avg_wait_before": round(df.loading_wait_min.mean(), 1), "avg_wait_after": round(d2.loading_wait_min.mean(), 1),
               "annual_cost_saving_rs": int((df.transport_cost - cost2).sum()),
               "cap_wait_min": round(offpeak_med + 4, 1)}

# rain buffer what-if: promised time + 20% on rain days
d3 = df.copy()
d3["sched3"] = np.where(d3.weather == "Rain", d3.scheduled_hrs * 1.2, d3.scheduled_hrs)
R["rain_buffer_late_pct"] = {"before": round(df.is_late.mean() * 100, 1),
                             "after": round((d3.delivery_hrs > d3.sched3).mean() * 100, 1),
                             "rain_late_before": round(df.loc[df.weather == "Rain", "is_late"].mean() * 100, 1),
                             "rain_late_after": round((d3.loc[d3.weather == "Rain", "delivery_hrs"] > d3.loc[d3.weather == "Rain", "sched3"]).mean() * 100, 1)}

with open(f"{OUT}/results3.json", "w") as fh:
    json.dump(R, fh, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
print(json.dumps(R, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))