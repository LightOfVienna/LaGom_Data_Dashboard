"""Lagom Track A - Fiscal Impact Sensitivity Dashboard (Streamlit).

Run locally from the project folder:   streamlit run dashboard/app.py
Deploy free: push the project (or just the dashboard/ folder with data/) to GitHub, then create an app on
Streamlit Community Cloud with main file dashboard/app.py.

Data: inside the project the app reads final_output/ (Modules 1-7) and refreshes dashboard/data/dashboard_counties.csv;
deployed on its own it reads that snapshot. The sensitivity math is Module 7's: every value scales with homes, county
property tax moves with price, school property tax is recalculated above the county's school homestead exemption,
and the school result follows Module 3's capacity cases.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
FINAL = APP_DIR.parent / "final_output"
SNAPSHOT = APP_DIR / "data" / "dashboard_counties.csv"
YEARS, PHASE_FACTOR = 10, 8.0          # 10-year horizon; 5-year phase-in 20/40/60/80% then 100% = 8 full years of tax
HOME_OPTIONS = {"100 homes": 100, "150 homes (standard)": 150, "200 homes": 200}
PRICE_MIN, PRICE_MAX, PRICE_STEP = 150_000, 500_000, 5_000
CASES = {"county": "County estimate", "under": "Under 85% of capacity", "near": "85-100% of capacity",
         "over": "Over capacity (about 110%)"}


# ------------------------------------------------------------------------------------------------ data
def build_data(final_dir):
    """Per-home values for the Tier 1 / Tier 2 counties from the Module 3 and Module 7 outputs."""
    m = pd.read_csv(final_dir / "Municipal_Value_Summary.csv", dtype={"county_fips": str})
    s = pd.read_csv(final_dir / "School_Impact.csv", dtype={"county_fips": str})
    p = pd.read_csv(final_dir / "Property_Tax_Revenue_full.csv", dtype={"county_fips": str}, usecols=["county_fips", "homes_modeled"])
    for d in (m, s, p):
        d["county_fips"] = d["county_fips"].str.zfill(5)
    s = s.set_index("county_fips")
    # unrounded home count (Module 1); Module 7's community_homes is rounded to 0.1, which skews per-home values slightly
    # (if Module 7 was run with a fixed COMMUNITY_HOMES, its own home count is used instead)
    modeled = m["county_fips"].map(p.set_index("county_fips")["homes_modeled"])
    h = modeled if np.allclose(modeled.round(1), m["community_homes"]) else m["community_homes"]
    out = m[["county_fips", "county_state", "state_abbr", "final_tier", "priority_rank", "fiscal_only_rank",
             "fiscal_impact_score", "market_score", "overall_county_rank", "community_homes", "avg_home_price",
             "assessment_ratio", "county_general_mill_rate", "school_mill_rate", "school_mill_rate_applied",
             "school_exemption_per_home", "county_retained_rate_pct", "pilot_rate", "capacity_flag", "main_district",
             "avg_weekly_wage", "owner_occupied_note"]].copy()
    out["county"] = out["county_state"].str.split(",").str[0]
    for c in ["county_property_tax", "county_sales_tax", "school_property_tax", "school_state_aid_net", "inspection_offset",
              "construction_year_total_jobs", "ongoing_jobs_supported", "ongoing_jobs_earnings", "new_students",
              "new_residents", "employed_residents_added"]:
        out[c + "_ph"] = m[c] / h
    for case, col in [("under", "net_impact_under85"), ("near", "net_impact_mid"), ("over", "net_impact_over100")]:
        out[f"school_aid_net_{case}_ph"] = (s.loc[out["county_fips"], col] / s.loc[out["county_fips"], "homes"]).values
    out["school_aid_net_county_ph"] = out["school_state_aid_net_ph"]
    return out


@st.cache_data
def load_data():
    if (FINAL / "Municipal_Value_Summary.csv").exists() and (FINAL / "School_Impact.csv").exists():
        df = build_data(FINAL)
        try:                                   # keep the snapshot current for deployment
            SNAPSHOT.parent.mkdir(exist_ok=True)
            df.to_csv(SNAPSHOT, index=False)
        except OSError:
            pass
        return df, "final_output/ (Modules 1-7)"
    if SNAPSHOT.exists():
        return pd.read_csv(SNAPSHOT, dtype={"county_fips": str}), "dashboard/data/dashboard_counties.csv"
    st.error("No data found: run Modules 1-7 first (final_output/), or add dashboard/data/dashboard_counties.csv.")
    st.stop()


# ------------------------------------------------------------------------------------------------ the model
def compute(d, homes, price=None, case="county"):
    """Fiscal impact for a community of `homes` at an average sale `price` (None = each county's own average),
    with the school result in capacity `case`. Works on one county (Series) or many (DataFrame)."""
    df = d.to_frame().T if isinstance(d, pd.Series) else d
    p = df["avg_home_price"].astype(float) if price is None else pd.Series(float(price), index=df.index)
    school_mills = df["school_mill_rate_applied"].astype(float).fillna(0)
    out = pd.DataFrame(index=df.index)
    out["homes"] = float(homes)
    out["avg_price"] = p
    out["county_property_tax"] = df["county_property_tax_ph"].astype(float) * p / df["avg_home_price"].astype(float) * homes
    out["county_sales_tax"] = df["county_sales_tax_ph"].astype(float) * homes
    out["school_property_tax"] = (np.maximum(p - df["school_exemption_per_home"].astype(float).fillna(0), 0)
                                  * df["assessment_ratio"].astype(float) * school_mills / 1000 * homes)
    out["school_state_aid_net"] = df[f"school_aid_net_{case}_ph"].astype(float) * homes
    out["county_revenue"] = out["county_property_tax"] + out["county_sales_tax"]
    out["school_net"] = out["school_property_tax"] + out["school_state_aid_net"]
    out["total_annual_value"] = out["county_revenue"] + out["school_net"]
    prop = out["county_property_tax"] + out["school_property_tax"]
    recurring = (out["county_sales_tax"] + out["school_state_aid_net"]) * YEARS
    out[f"total_{YEARS}yr_no_abatement"] = prop * YEARS + recurring
    out[f"total_{YEARS}yr_phase_in"] = prop * PHASE_FACTOR + recurring
    out[f"total_{YEARS}yr_pilot"] = prop * df["pilot_rate"].astype(float) * YEARS + recurring
    for c in ["inspection_offset", "construction_year_total_jobs", "ongoing_jobs_supported", "ongoing_jobs_earnings",
              "new_students", "new_residents", "employed_residents_added"]:
        out[c] = df[c + "_ph"].astype(float) * homes
    return out.iloc[0] if isinstance(d, pd.Series) else out


def usd(x, short=False):
    if pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if short and x >= 1e6:
        return f"{sign}${x / 1e6:,.2f}M"
    if short and x >= 1e4:
        return f"{sign}${x / 1e3:,.0f}K"
    return f"{sign}${x:,.0f}"


# ------------------------------------------------------------------------------------------------ page
st.set_page_config(page_title="Lagom Fiscal Impact Dashboard", layout="wide")
data, source = load_data()

with st.sidebar:
    st.header("County and scenario")
    states = sorted(data["state_abbr"].unique())
    default_state = data.sort_values("priority_rank").iloc[0]["state_abbr"]
    state = st.selectbox("State", states, index=states.index(default_state))
    in_state = data[data["state_abbr"] == state].sort_values("priority_rank")
    labels = [f"{r.county} (priority rank {int(r.priority_rank)})" for r in in_state.itertuples()]
    pick = st.selectbox("County", labels, index=0)
    row = in_state.iloc[labels.index(pick)]

    size_label = st.radio("Sensitivity: community size", list(HOME_OPTIONS), index=1)
    homes = HOME_OPTIONS[size_label]

    own_price = float(row["avg_home_price"])
    default_price = int(min(max(round(own_price / PRICE_STEP) * PRICE_STEP, PRICE_MIN), PRICE_MAX))
    price = st.slider("Average sales price", PRICE_MIN, PRICE_MAX, default_price, PRICE_STEP, format="$%d")
    if price == default_price:
        price = own_price                  # slider left at the default: use the county's exact average
    st.caption(f"County average: {usd(own_price)}. Syllabus sensitivity: $220,000 vs $280,000.")

    case_labels = {k: (f"{v}: {row['capacity_flag']}" if k == "county" else v) for k, v in CASES.items()}
    case = st.selectbox("School district capacity", list(CASES), format_func=lambda k: case_labels[k])
    st.divider()
    st.caption(f"Data: {source}. {len(data)} Tier 1 / Tier 2 counties.")

base = compute(row, 150, own_price, "county")            # reference: standard community, county price and capacity
cur = compute(row, homes, price, case)

st.title("Lagom Fiscal Impact Dashboard")
st.subheader(f"{row['county_state']}")
st.caption(f"{row['final_tier']} · priority rank {int(row['priority_rank'])} of {len(data)} · fiscal impact score "
           f"{row['fiscal_impact_score']:.0f} · market score {row['market_score']:.0f} · "
           f"{homes} homes at {usd(price)} · school case: {case_labels[case]}")

k1, k2, k3, k4 = st.columns(4)
for col, label, key in [(k1, "Total annual value", "total_annual_value"), (k2, "County government", "county_revenue"),
                        (k3, "School district", "school_net"), (k4, f"{YEARS}-year value (5-yr phase-in)", f"total_{YEARS}yr_phase_in")]:
    delta = cur[key] - base[key]
    col.metric(label, usd(cur[key], short=True), None if abs(delta) < 1 else f"{usd(delta, short=True)} vs base")
st.caption("Base = 150 homes at the county's average price with the county's school capacity estimate.")

tab_sum, tab_sens, tab_cmp, tab_about = st.tabs(["Summary", "Sensitivity", "Compare counties", "About"])

with tab_sum:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Annual value to local government**")
        lines = pd.DataFrame({
            "Source of value": ["Property tax: county", "Sales tax kept by the county", "County government subtotal",
                                "Property tax: schools", "School funding: state aid minus cost of new students",
                                "School district subtotal", "Total annual value"],
            "Per year": [cur["county_property_tax"], cur["county_sales_tax"], cur["county_revenue"], cur["school_property_tax"],
                         cur["school_state_aid_net"], cur["school_net"], cur["total_annual_value"]]})
        lines["Per home"] = lines["Per year"] / homes
        st.dataframe(lines.style.format({"Per year": usd, "Per home": usd}), hide_index=True)
        st.bar_chart(lines.iloc[[0, 1, 3, 4]].set_index("Source of value")["Per year"], height=260)
    with right:
        st.markdown(f"**{YEARS}-year value by property tax option**")
        st.dataframe(pd.DataFrame({"Option": ["No abatement", "5-year phase-in", f"PILOT at {row['pilot_rate']:.0%}"],
                                   "Value": [cur[f"total_{YEARS}yr_no_abatement"], cur[f"total_{YEARS}yr_phase_in"],
                                             cur[f"total_{YEARS}yr_pilot"]]}).style.format({"Value": usd}),
                     hide_index=True)
        st.markdown("**One-time and people**")
        st.dataframe(pd.DataFrame({
            "Measure": ["Building department savings (one-time)", "Jobs in the construction year", "Local jobs supported each year",
                        "Yearly earnings of those jobs", "New residents", "Residents working", "New students"],
            "Value": [usd(cur["inspection_offset"]), f"{cur['construction_year_total_jobs']:.0f}",
                      f"{cur['ongoing_jobs_supported']:.0f}", usd(cur["ongoing_jobs_earnings"]), f"{cur['new_residents']:.0f}",
                      f"{cur['employed_residents_added']:.0f}", f"{cur['new_students']:.0f}"]}),
            hide_index=True)
        st.markdown("**County rates used**")
        school_txt = ("no separate school levy (schools funded through the county tax)" if pd.isna(row["school_mill_rate"])
                      else f"{row['school_mill_rate_applied']:.2f} school mills applied")
        st.caption(f"{row['county_general_mill_rate']:.2f} county mills, {school_txt}, assessment ratio {row['assessment_ratio']:.0%}; "
                   f"county sales tax {row['county_retained_rate_pct']:.2f}%; average weekly wage {usd(row['avg_weekly_wage'])}."
                   + (f" Owner-occupied rule: {row['owner_occupied_note']}." if isinstance(row["owner_occupied_note"], str)
                      and row["owner_occupied_note"] else ""))

with tab_sens:
    st.markdown(f"**Total annual value by community size and school capacity** (at {usd(price)})")
    grid = pd.DataFrame({CASES[c] if c != "county" else f"County estimate ({row['capacity_flag']})":
                         [compute(row, h, price, c)["total_annual_value"] for h in HOME_OPTIONS.values()] for c in CASES},
                        index=list(HOME_OPTIONS))
    st.dataframe(grid.style.format(usd))
    st.bar_chart(grid, height=300)
    st.markdown(f"**Total annual value by average sales price** ({homes} homes, {CASES[case].lower()})")
    prices = np.arange(PRICE_MIN, PRICE_MAX + 1, 10_000)
    curve = pd.DataFrame({"Average sales price": prices,
                          "Total annual value": [compute(row, homes, p, case)["total_annual_value"] for p in prices],
                          "County government": [compute(row, homes, p, case)["county_revenue"] for p in prices],
                          "School district": [compute(row, homes, p, case)["school_net"] for p in prices]})
    st.line_chart(curve.set_index("Average sales price"), height=300)
    syl = pd.DataFrame([{"Scenario": label, "Total annual value": compute(row, h, p, c)["total_annual_value"]}
                        for label, h, p, c in [("Base (150 homes, county price, county capacity)", 150, own_price, "county"),
                                               ("100 homes", 100, own_price, "county"), ("200 homes", 200, own_price, "county"),
                                               ("$220K average price", 150, 220_000, "county"),
                                               ("$280K average price", 150, 280_000, "county"),
                                               ("School at 85% of capacity", 150, own_price, "under"),
                                               ("School at 110% of capacity", 150, own_price, "over")]])
    st.markdown("**Syllabus sensitivity cases** (one change at a time from the base)")
    st.dataframe(syl.style.format({"Total annual value": usd}), hide_index=True)

with tab_cmp:
    scope = st.radio("Counties", [f"All {state} Tier 1 / Tier 2 counties", "All 434 Tier 1 / Tier 2 counties"], horizontal=True)
    same_price = st.checkbox(f"Use {usd(price)} for every county (otherwise each county's own average price)", value=False)
    pool = in_state if scope.startswith("All " + state) else data.sort_values("priority_rank")
    res = compute(pool, homes, price if same_price else None, case)
    table = pd.DataFrame({"Priority rank": pool["priority_rank"].astype(int), "County": pool["county_state"],
                          "Tier": pool["final_tier"].str.split(" - ").str[0], "Avg price": res["avg_price"],
                          "Total annual value": res["total_annual_value"], "County government": res["county_revenue"],
                          "School district": res["school_net"], f"{YEARS}-yr (phase-in)": res[f"total_{YEARS}yr_phase_in"],
                          "Capacity estimate": pool["capacity_flag"]})
    money = ["Avg price", "Total annual value", "County government", "School district", f"{YEARS}-yr (phase-in)"]
    st.dataframe(table.style.format({c: usd for c in money}), hide_index=True, height=480)
    st.download_button("Download this table (CSV)", table.to_csv(index=False).encode(), "lagom_county_comparison.csv", "text/csv")

with tab_about:
    st.markdown(f"""
**What this shows.** The estimated yearly value of a Lagom community to local government in a Tier 1 / Tier 2 county, from the Track A model (Modules 1-7): county property tax, the county's share of sales tax, school property tax, and state school aid minus the cost of the new students.

**How the toggles work** (the same math as Module 7's sensitivity analysis):
- *Community size:* every value scales with the number of homes.
- *Average sales price:* county property tax moves in proportion to price. School property tax is recalculated on the price above the county's school homestead exemption (TX $140,000, LA $75,000, FL $25,000; SC homeowners pay school debt-service millage only). Sales tax, schools and jobs do not depend on price.
- *School capacity:* under 85% of capacity, 90% of the per-pupil cost is avoided; 85-100%, half; over capacity, none (Module 3). "County estimate" uses the county's own case, which for most counties is estimated from enrollment growth.

**Limits.** Estimates in today's dollars, before growth in values or rates. Georgia figures are before homestead exemptions. Jobs and inspection savings scale with homes only. Data source: {source}.
""")
