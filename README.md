# Lagom Fiscal Impact Dashboard (Streamlit)

An interactive sensitivity dashboard for the Track A model. It covers the syllabus Key Deliverable "Sensitivity Analysis" (p. 5).

**Controls (sidebar)**
- **State and county:** any of the 434 Tier 1 / Tier 2 counties. Counties are listed by priority rank.
- **Community size:** 100, 150 (standard) or 200 homes.
- **Average sales price:** $150,000 to $500,000. The default is the county's own average.
- **School district capacity:** the county's estimate, under 85%, 85–100%, or over capacity (about 110%).

**Tabs**
- **Summary:** yearly value by line item, 10-year value under each property tax option, jobs and residents, and the county's rates.
- **Sensitivity:** home count by capacity, a price curve, and the syllabus cases.
- **Compare counties:** the state or all 434 counties under the same settings, downloadable as CSV.
- **About:** the method behind the numbers.

The math is the same as Module 7's sensitivity analysis. The app reproduces `Sensitivity_Analysis.csv` to within a dollar, except the 100- and 200-home cases, which can differ by up to about $60 because Module 7's per-home figures were rounded.

## Run it on your computer

From the project folder:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Inside the project, the app reads `final_output/` directly, so re-running Modules 1–7 updates it. It also refreshes `dashboard/data/dashboard_counties.csv`, a small snapshot of the data it needs.

## Put it online for free (Streamlit Community Cloud)

1. Push the project to a GitHub repository; it can be private. If the data folders are too large for GitHub, push only the `dashboard/` folder. The app then runs from `dashboard/data/dashboard_counties.csv`.
2. Sign in at share.streamlit.io with GitHub, choose **Create app**, pick the repository, and set the main file to `dashboard/app.py` (or `app.py` if you pushed only the dashboard folder).
3. Deploy. To update the numbers later, re-run Modules 1–7, run the app once locally to refresh the snapshot, and push again.

The app only shows Tier 1 / Tier 2 counties, the counties Modules 4 and 7 cover.
