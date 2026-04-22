import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Miller Construction – Bid Simulator", layout="wide")

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("project_costs.csv")
    df.columns = ["bid_prep", "completion"]
    df["bid_prep"] = pd.to_numeric(df["bid_prep"], errors="coerce")
    df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
    bid_prep = df["bid_prep"].dropna().values
    completion = np.maximum(df["completion"].dropna().values, 70_000)
    return bid_prep, completion

bid_prep_data, completion_data = load_data()

# ── Distribution helpers ──────────────────────────────────────────────────────

DIST_OPTIONS = ["Empirical", "Normal", "Lognormal", "Uniform", "Triangular"]
COMPETITOR_DIST_OPTIONS = ["Triangular", "Normal", "Uniform"]

def auto_fit(data, dist):
    if dist == "Normal":
        return {"mean": float(np.mean(data)), "std": float(np.std(data))}
    if dist == "Lognormal":
        log_data = np.log(data)
        return {"log_mean": float(np.mean(log_data)), "log_std": float(np.std(log_data))}
    if dist == "Uniform":
        return {"low": float(np.min(data)), "high": float(np.max(data))}
    if dist == "Triangular":
        return {"tri_min": float(np.min(data)), "tri_mode": float(np.median(data)), "tri_max": float(np.max(data))}
    return {}


def draw_samples(rng, dist, params, data, n):
    if dist == "Empirical":
        return rng.choice(data, size=n, replace=True)
    if dist == "Normal":
        return rng.normal(params["mean"], params["std"], size=n)
    if dist == "Lognormal":
        return np.exp(rng.normal(params["log_mean"], params["log_std"], size=n))
    if dist == "Uniform":
        return rng.uniform(params["low"], params["high"], size=n)
    if dist == "Triangular":
        return rng.triangular(params["tri_min"], params["tri_mode"], params["tri_max"], size=n)
    raise ValueError(f"Unknown distribution: {dist}")


def dist_pdf(x, dist, params, data):
    if dist == "Empirical":
        return stats.gaussian_kde(data)(x)
    if dist == "Normal":
        return stats.norm.pdf(x, params["mean"], params["std"])
    if dist == "Lognormal":
        return stats.lognorm.pdf(x, s=params["log_std"], scale=np.exp(params["log_mean"]))
    if dist == "Uniform":
        return stats.uniform.pdf(x, params["low"], params["high"] - params["low"])
    if dist == "Triangular":
        c = (params["tri_mode"] - params["tri_min"]) / (params["tri_max"] - params["tri_min"])
        return stats.triang.pdf(x, c=c, loc=params["tri_min"], scale=params["tri_max"] - params["tri_min"])
    return np.zeros_like(x)


# ── Distribution param UI helper ─────────────────────────────────────────────

def dist_param_ui(label, dist, defaults, key_prefix):
    params = {}
    if dist == "Normal":
        params["mean"] = st.sidebar.number_input(f"{label} mean ($)", value=defaults["mean"], step=100.0, key=f"{key_prefix}_mean")
        params["std"]  = st.sidebar.number_input(f"{label} std dev ($)", value=defaults["std"], step=100.0, min_value=1.0, key=f"{key_prefix}_std")
    elif dist == "Lognormal":
        params["log_mean"] = st.sidebar.number_input(f"{label} log-mean", value=defaults["log_mean"], step=0.01, key=f"{key_prefix}_lm")
        params["log_std"]  = st.sidebar.number_input(f"{label} log-std",  value=defaults["log_std"],  step=0.01, min_value=0.001, key=f"{key_prefix}_ls")
    elif dist == "Uniform":
        params["low"]  = st.sidebar.number_input(f"{label} min ($)", value=defaults["low"],  step=100.0, key=f"{key_prefix}_low")
        params["high"] = st.sidebar.number_input(f"{label} max ($)", value=defaults["high"], step=100.0, key=f"{key_prefix}_high")
    elif dist == "Triangular":
        params["tri_min"]  = st.sidebar.number_input(f"{label} min ($)",  value=defaults["tri_min"],  step=100.0, key=f"{key_prefix}_tmin")
        params["tri_mode"] = st.sidebar.number_input(f"{label} mode ($)", value=defaults["tri_mode"], step=100.0, key=f"{key_prefix}_tmode")
        params["tri_max"]  = st.sidebar.number_input(f"{label} max ($)",  value=defaults["tri_max"],  step=100.0, key=f"{key_prefix}_tmax")
    return params


# ── Simulation core ──────────────────────────────────────────────────────────

def run_simulation(miller_bid, n, seed, prep_dist, prep_params,
                   comp_dist, comp_params, comp_bid_dist, comp_bid_params,
                   competitor_prob):
    rng = np.random.default_rng(seed)

    prep_costs = draw_samples(rng, prep_dist, prep_params, bid_prep_data, n)
    comp_costs = np.maximum(draw_samples(rng, comp_dist, comp_params, completion_data, n), 70_000)

    def rival_bids(size):
        return draw_samples(rng, comp_bid_dist, comp_bid_params, np.array([]), size)

    bids_a       = rival_bids(n)
    b_enters     = rng.random(n) < competitor_prob
    c_enters     = rng.random(n) < competitor_prob
    bids_b_raw   = rival_bids(n)          # always drawn; masked below
    bids_c_raw   = rival_bids(n)
    bids_b       = np.where(b_enters, bids_b_raw, np.inf)
    bids_c       = np.where(c_enters, bids_c_raw, np.inf)

    min_competitor = np.minimum(np.minimum(bids_a, bids_b), bids_c)
    won   = miller_bid < min_competitor
    profit = np.where(won, miller_bid - comp_costs - prep_costs, -prep_costs)

    return {
        "profits":       profit,
        "won":           won,
        "prep_costs":    prep_costs,
        "comp_costs":    comp_costs,
        "n_competitors": b_enters.astype(int) + c_enters.astype(int) + 1,
        "bids_a":        bids_a,
        "bids_b_raw":    bids_b_raw,
        "bids_c_raw":    bids_c_raw,
        "b_enters":      b_enters,
        "c_enters":      c_enters,
        "min_competitor": min_competitor,
    }


@st.cache_data
def sweep_bids(bid_min, bid_max, bid_step, n, seed,
               prep_dist, prep_params_frozen,
               comp_dist, comp_params_frozen,
               comp_bid_dist, comp_bid_params_frozen,
               competitor_prob):
    prep_params     = dict(prep_params_frozen)
    comp_params     = dict(comp_params_frozen)
    comp_bid_params = dict(comp_bid_params_frozen)

    bids = np.arange(bid_min, bid_max + bid_step, bid_step)
    rows = []
    for b in bids:
        res = run_simulation(b, n, seed, prep_dist, prep_params,
                             comp_dist, comp_params, comp_bid_dist, comp_bid_params,
                             competitor_prob)
        p = res["profits"]
        rows.append({
            "bid":       b,
            "e_profit":  np.mean(p),
            "p_win":     np.mean(res["won"]),
            "p_positive": np.mean(p > 0),
            "p5":        np.percentile(p, 5),
            "p95":       np.percentile(p, 95),
        })
    return pd.DataFrame(rows)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Simulation Settings")
n_sim = st.sidebar.select_slider(
    "Number of simulations", options=[10_000, 50_000, 100_000, 200_000], value=100_000
)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# -- Analysis mode
st.sidebar.markdown("---")
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode", ["Single Bid Analysis", "Bid Price Sweep"], index=1, key="analysis_mode"
)

if analysis_mode == "Single Bid Analysis":
    st.sidebar.subheader("Bid Price")
    single_bid = st.sidebar.number_input("Miller's bid price ($)", value=150_000, step=1_000)
    sweep_min = sweep_max = sweep_step = None
else:
    st.sidebar.subheader("Bid Price Sweep")
    sweep_min  = st.sidebar.number_input("Min bid ($)",  value=100_000, step=5_000)
    sweep_max  = st.sidebar.number_input("Max bid ($)",  value=200_000, step=5_000)
    sweep_step = st.sidebar.number_input("Step ($)",     value=500,     step=100)
    single_bid = None

# -- Bid prep distribution
st.sidebar.markdown("---")
st.sidebar.subheader("Bid Prep Cost Distribution")
prep_dist     = st.sidebar.selectbox("Distribution", DIST_OPTIONS, index=0, key="prep_dist")
prep_defaults = auto_fit(bid_prep_data, prep_dist)
prep_params   = dist_param_ui("Bid prep", prep_dist, prep_defaults, "prep")

# -- Completion cost distribution
st.sidebar.markdown("---")
st.sidebar.subheader("Completion Cost Distribution")
comp_dist     = st.sidebar.selectbox("Distribution", DIST_OPTIONS, index=0, key="comp_dist")
comp_defaults = auto_fit(completion_data, comp_dist)
comp_params   = dist_param_ui("Completion", comp_dist, comp_defaults, "comp")

# -- Competitor bid distribution
st.sidebar.markdown("---")
st.sidebar.subheader("Competitor Bid Distribution")
comp_bid_dist = st.sidebar.selectbox("Distribution", COMPETITOR_DIST_OPTIONS, index=0, key="comp_bid_dist")
_comp_bid_defaults = {
    "Triangular": {"tri_min": 90_000.0, "tri_mode": 130_000.0, "tri_max": 180_000.0},
    "Normal":     {"mean": 130_000.0, "std": 25_000.0},
    "Uniform":    {"low": 90_000.0, "high": 180_000.0},
}[comp_bid_dist]
comp_bid_params = dist_param_ui("Competitor bid", comp_bid_dist, _comp_bid_defaults, "cbid")

# -- Competitor entry probability
st.sidebar.markdown("---")
st.sidebar.subheader("Competitor Entry Probability")
competitor_prob = st.sidebar.slider(
    "Probability B & C each enter (%)", min_value=0, max_value=100, value=50, step=5
) / 100.0

# -- Run button
st.sidebar.markdown("---")
run_clicked = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("Miller Construction Co. — Bid Decision Simulator")
st.markdown("Monte Carlo simulation to evaluate whether to submit a proposal and at what price.")

tab2, tab1 = st.tabs(["Analysis Results", "Data Overview"])

# ── Shared fitted-PDF overlay helper ─────────────────────────────────────────

def hist_with_pdf(data, dist, params, color, title, x_label):
    x     = np.linspace(np.min(data) * 0.9, np.max(data) * 1.1, 400)
    y_pdf = dist_pdf(x, dist, params, data)
    fig   = go.Figure()
    fig.add_trace(go.Histogram(x=data, histnorm="probability density", nbinsx=25,
                               marker_color=color, opacity=0.6, name="Data"))
    fig.add_trace(go.Scatter(x=x, y=y_pdf, mode="lines",
                             line=dict(color="#1E293B", width=2), name=f"{dist} fit"))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Density",
                      height=300, legend=dict(orientation="h", y=1.05))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bid Preparation Costs")
        m1, m2, m3 = st.columns(3)
        m1.metric("n", len(bid_prep_data))
        m2.metric("Mean", f"${np.mean(bid_prep_data):,.0f}")
        m3.metric("Std Dev", f"${np.std(bid_prep_data):,.0f}")
        st.plotly_chart(hist_with_pdf(bid_prep_data, prep_dist, prep_params,
                                      "#2563EB", f"Bid Prep — {prep_dist}", "Cost ($)"),
                        use_container_width=True)

    with col2:
        st.subheader("Project Completion Costs")
        m4, m5, m6 = st.columns(3)
        m4.metric("n", len(completion_data))
        m5.metric("Mean", f"${np.mean(completion_data):,.0f}")
        m6.metric("Std Dev", f"${np.std(completion_data):,.0f}")
        st.plotly_chart(hist_with_pdf(completion_data, comp_dist, comp_params,
                                      "#16A34A", f"Completion Cost — {comp_dist}", "Cost ($)"),
                        use_container_width=True)

    st.markdown("---")
    st.subheader(f"Competitor Bid Distribution — {comp_bid_dist}")
    _lo = comp_bid_params.get("low",     comp_bid_params.get("tri_min",  comp_bid_params.get("mean", 130_000) - 3 * comp_bid_params.get("std", 25_000))) * 0.9
    _hi = comp_bid_params.get("high",    comp_bid_params.get("tri_max",  comp_bid_params.get("mean", 130_000) + 3 * comp_bid_params.get("std", 25_000))) * 1.1
    x_comp  = np.linspace(_lo, _hi, 500)
    y_comp  = dist_pdf(x_comp, comp_bid_dist, comp_bid_params, np.array([]))
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=x_comp, y=y_comp, fill="tozeroy",
                                  line=dict(color="#DC2626", width=2), name="Competitor bid PDF"))
    fig_comp.update_layout(xaxis_title="Bid Amount ($)", yaxis_title="Density", height=280)
    st.plotly_chart(fig_comp, use_container_width=True)


# ── Run simulation on button click ───────────────────────────────────────────

if run_clicked:
    if analysis_mode == "Single Bid Analysis":
        with st.spinner(f"Running {n_sim:,} simulations for bid ${single_bid:,}…"):
            st.session_state.sim_res = run_simulation(
                single_bid, n_sim, seed,
                prep_dist, prep_params, comp_dist, comp_params,
                comp_bid_dist, comp_bid_params, competitor_prob,
            )
        st.session_state.sim_single_bid = single_bid
        st.session_state.sim_sweep_df   = None
    else:
        with st.spinner(f"Sweeping bid prices from ${sweep_min:,} to ${sweep_max:,}…"):
            st.session_state.sim_sweep_df = sweep_bids(
                sweep_min, sweep_max, sweep_step, n_sim, seed,
                prep_dist, tuple(sorted(prep_params.items())),
                comp_dist, tuple(sorted(comp_params.items())),
                comp_bid_dist, tuple(sorted(comp_bid_params.items())),
                competitor_prob,
            )
        st.session_state.sim_res        = None
        st.session_state.sim_single_bid = None


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis Results
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    if "sim_res" not in st.session_state and "sim_sweep_df" not in st.session_state:
        st.info("Configure your settings in the sidebar, then click **▶ Run Simulation**.")
        st.stop()

    # ── Single Bid Analysis ───────────────────────────────────────────────────
    if analysis_mode == "Single Bid Analysis":
        if st.session_state.get("sim_res") is None:
            st.info("Click **▶ Run Simulation** to see results.")
            st.stop()

        res        = st.session_state.sim_res
        profits    = res["profits"]
        won        = res["won"]
        single_bid = st.session_state.sim_single_bid

        # ── Summary metrics
        st.subheader(f"Results for Miller's Bid = ${single_bid:,}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("E[Profit]",       f"${np.mean(profits):,.0f}")
        m2.metric("P(Win)",          f"{np.mean(won):.1%}")
        m3.metric("P(Profit > 0)",   f"{np.mean(profits > 0):.1%}")
        m4.metric("5th Percentile",  f"${np.percentile(profits, 5):,.0f}")
        m5.metric("95th Percentile", f"${np.percentile(profits, 95):,.0f}")

        st.markdown("---")

        # ── Histogram + Conditional stats
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Profit Distribution")
            hist_bins = st.slider("Number of bins", min_value=10, max_value=200, value=60, step=5, key="hist_bins")
            won_profits  = profits[won]
            lost_profits = profits[~won]
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(x=profits[profits >= 0], nbinsx=hist_bins,
                                        marker_color="#16A34A", name="Profit ≥ 0", opacity=0.8))
            fig4.add_trace(go.Histogram(x=profits[profits < 0],  nbinsx=max(hist_bins // 4, 5),
                                        marker_color="#DC2626", name="Profit < 0", opacity=0.8))
            fig4.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
            fig4.add_vline(x=np.mean(profits), line_dash="dot", line_color="#2563EB",
                           annotation_text=f"Mean: ${np.mean(profits):,.0f}",
                           annotation_position="top right")
            fig4.update_layout(barmode="overlay", xaxis_title="Net Profit ($)", yaxis_title="Count",
                               height=380, legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig4, use_container_width=True)

        with col_b:
            st.subheader("Conditional Stats")
            st.markdown("**When Miller wins:**")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("E[Profit | Win]",      f"${np.mean(won_profits):,.0f}"       if len(won_profits) else "N/A")
            wc2.metric("P(Profit > 0 | Win)",  f"{np.mean(won_profits > 0):.1%}"     if len(won_profits) else "N/A")
            wc3.metric("Trials won",           f"{len(won_profits):,}")

            st.markdown("**When Miller loses:**")
            lc1, lc2 = st.columns(2)
            lc1.metric("E[Loss | Lose]", f"${np.mean(lost_profits):,.0f}" if len(lost_profits) else "N/A")
            lc2.metric("Trials lost",    f"{len(lost_profits):,}")

            st.markdown("---")
            st.subheader("Number of Competitors")
            n_comp_counts = pd.Series(res["n_competitors"]).value_counts().sort_index()
            fig5 = px.bar(x=n_comp_counts.index, y=n_comp_counts.values / n_sim,
                          labels={"x": "Number of Competitors", "y": "Probability"},
                          color_discrete_sequence=["#7C3AED"], text_auto=".1%")
            fig5.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("---")

        # ── CDF + Pie chart
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Cumulative Distribution (CDF)")
            sorted_profits = np.sort(profits)
            cdf = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(x=sorted_profits, y=cdf, mode="lines",
                                         line=dict(color="#2563EB", width=2), name="CDF"))
            fig_cdf.add_vline(x=0, line_dash="dash", line_color="#DC2626", line_width=1,
                              annotation_text="Break-even", annotation_position="top right")
            p_above_zero = float(np.mean(profits > 0))
            fig_cdf.add_hline(y=1 - p_above_zero, line_dash="dot", line_color="#6B7280", line_width=1)
            fig_cdf.update_layout(xaxis_title="Net Profit ($)", yaxis_title="Cumulative Probability",
                                  height=380, legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_cdf, use_container_width=True)

        with col_d:
            st.subheader("Outcome Breakdown")
            n_win_profit  = int(np.sum(won  & (profits > 0)))
            n_win_loss    = int(np.sum(won  & (profits <= 0)))
            n_lost        = int(np.sum(~won))
            fig_pie = go.Figure(go.Pie(
                labels=["Won & Profitable", "Won & Unprofitable", "Lost"],
                values=[n_win_profit, n_win_loss, n_lost],
                marker=dict(colors=["#16A34A", "#F59E0B", "#DC2626"]),
                textinfo="label+percent",
                hole=0.35,
            ))
            fig_pie.update_layout(height=380, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # ── Cost Breakdown box plots
        st.subheader("Cost Breakdown (winning trials only)")
        if len(won_profits) > 0:
            df_win = pd.DataFrame({
                "Completion Cost": res["comp_costs"][won],
                "Bid Prep Cost":   res["prep_costs"][won],
                "Net Profit":      won_profits,
            })
            fig6 = go.Figure()
            for col, color in zip(["Completion Cost", "Bid Prep Cost", "Net Profit"],
                                  ["#F59E0B", "#3B82F6", "#10B981"]):
                fig6.add_trace(go.Box(y=df_win[col], name=col, marker_color=color, boxmean=True))
            fig6.update_layout(yaxis_title="Amount ($)", height=350)
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown("---")

        # ── Sample scenarios table
        st.subheader("Sample Scenarios")
        n_scenarios = st.slider("Number of scenarios to show", min_value=5, max_value=500,
                                value=20, step=5, key="n_scenarios")
        idx = np.arange(min(n_scenarios, n_sim))

        def fmt_bid(entered, bid_val):
            return np.where(entered, bid_val, np.nan)

        b_bids = fmt_bid(res["b_enters"][idx], res["bids_b_raw"][idx])
        c_bids = fmt_bid(res["c_enters"][idx], res["bids_c_raw"][idx])

        scenario_df = pd.DataFrame({
            "Scenario #":       idx + 1,
            "Bid Prep Cost":    res["prep_costs"][idx],
            "Project Cost":     res["comp_costs"][idx],
            "Comp A Bid":       res["bids_a"][idx],
            "Comp B Bid":       b_bids,
            "Comp C Bid":       c_bids,
            "Lowest Comp Bid":  res["min_competitor"][idx],
            "Miller Wins?":     np.where(res["won"][idx], "✅ Yes", "❌ No"),
            "Net Profit":       res["profits"][idx],
        })

        fmt_cols = ["Bid Prep Cost", "Project Cost", "Comp A Bid", "Comp B Bid",
                    "Comp C Bid", "Lowest Comp Bid", "Net Profit"]
        for c in fmt_cols:
            scenario_df[c] = scenario_df[c].apply(
                lambda v: "Did not bid" if pd.isna(v) else f"${v:,.0f}"
            )

        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # ── Bid Price Sweep ───────────────────────────────────────────────────────
    else:
        if st.session_state.get("sim_sweep_df") is None:
            st.info("Click **▶ Run Simulation** to see results.")
            st.stop()

        sweep_df    = st.session_state.sim_sweep_df
        optimal_row = sweep_df.loc[sweep_df["e_profit"].idxmax()]
        optimal_bid    = optimal_row["bid"]
        optimal_profit = optimal_row["e_profit"]

        st.subheader("Optimal Bid Analysis")
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Optimal Bid Price",        f"${optimal_bid:,.0f}")
        oc2.metric("Max E[Profit]",            f"${optimal_profit:,.0f}")
        oc3.metric("Win Prob at Optimal",      f"{optimal_row['p_win']:.1%}")
        oc4.metric("P(Profit > 0) at Optimal", f"{optimal_row['p_positive']:.1%}")

        st.markdown("---")

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["p95"], fill=None, mode="lines",
                                  line_color="rgba(37,99,235,0.2)", showlegend=False))
        fig7.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["p5"], fill="tonexty", mode="lines",
                                  line_color="rgba(37,99,235,0.2)", fillcolor="rgba(37,99,235,0.12)",
                                  name="5th–95th pct band"))
        fig7.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["e_profit"],
                                  mode="lines+markers", line=dict(color="#2563EB", width=2.5),
                                  name="E[Profit]"))
        fig7.add_vline(x=optimal_bid, line_dash="dash", line_color="#DC2626",
                       annotation_text=f"Optimal: ${optimal_bid:,}", annotation_position="top left")
        fig7.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
        fig7.update_layout(title="Expected Profit vs. Bid Price",
                           xaxis_title="Bid Price ($)", yaxis_title="Net Profit ($)",
                           height=800, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig7, use_container_width=True)

        col_p, col_q = st.columns(2)

        with col_p:
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["p_win"],
                                      mode="lines+markers", line=dict(color="#16A34A", width=2.5), name="P(Win)"))
            fig8.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["p_positive"],
                                      mode="lines+markers", line=dict(color="#7C3AED", width=2.5), name="P(Profit > 0)"))
            fig8.add_vline(x=optimal_bid, line_dash="dash", line_color="#DC2626")
            fig8.update_layout(title="Win Probability vs. Bid Price",
                               xaxis_title="Bid Price ($)", yaxis_title="Probability",
                               height=350, legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig8, use_container_width=True)

        with col_q:
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=sweep_df["bid"], y=sweep_df["e_profit"] * sweep_df["p_win"],
                                      mode="lines+markers", line=dict(color="#F59E0B", width=2.5),
                                      name="E[Profit] × P(Win)"))
            fig9.add_vline(x=optimal_bid, line_dash="dash", line_color="#DC2626")
            fig9.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
            fig9.update_layout(title="Risk-Weighted Profit",
                               xaxis_title="Bid Price ($)", yaxis_title="($)", height=350)
            st.plotly_chart(fig9, use_container_width=True)

        st.markdown("---")
        st.subheader("Full Sweep Table")
        display_df = sweep_df.copy()
        display_df.columns = ["Bid Price", "E[Profit]", "P(Win)", "P(Profit>0)", "5th Pct", "95th Pct"]
        for col in ["Bid Price", "E[Profit]", "5th Pct", "95th Pct"]:
            display_df[col] = display_df[col].map("${:,.0f}".format)
        for col in ["P(Win)", "P(Profit>0)"]:
            display_df[col] = display_df[col].map("{:.1%}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Detailed drill-down at a selected bid price ───────────────────────
        st.markdown("---")
        st.subheader("Detailed Analysis at Selected Bid Price")

        available_bids = sweep_df["bid"].values
        default_idx    = int(np.argmax(sweep_df["e_profit"].values))
        selected_bid   = st.selectbox(
            "Select bid price",
            options=available_bids,
            index=default_idx,
            format_func=lambda x: f"${x:,.0f}",
            key="sweep_detail_bid",
        )

        d = run_simulation(
            selected_bid, n_sim, seed,
            prep_dist, prep_params, comp_dist, comp_params,
            comp_bid_dist, comp_bid_params, competitor_prob,
        )
        d_profits      = d["profits"]
        d_won          = d["won"]
        d_won_profits  = d_profits[d_won]
        d_lost_profits = d_profits[~d_won]

        dm1, dm2, dm3, dm4, dm5 = st.columns(5)
        dm1.metric("E[Profit]",       f"${np.mean(d_profits):,.0f}")
        dm2.metric("P(Win)",          f"{np.mean(d_won):.1%}")
        dm3.metric("P(Profit > 0)",   f"{np.mean(d_profits > 0):.1%}")
        dm4.metric("5th Percentile",  f"${np.percentile(d_profits, 5):,.0f}")
        dm5.metric("95th Percentile", f"${np.percentile(d_profits, 95):,.0f}")

        st.markdown("")

        # Histogram + conditional stats
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Profit Distribution**")
            d_hist_bins = st.slider("Number of bins", min_value=10, max_value=200,
                                    value=60, step=5, key="sweep_hist_bins")
            fig_dh = go.Figure()
            fig_dh.add_trace(go.Histogram(x=d_profits[d_profits >= 0], nbinsx=d_hist_bins,
                                           marker_color="#16A34A", name="Profit ≥ 0", opacity=0.8))
            fig_dh.add_trace(go.Histogram(x=d_profits[d_profits < 0],
                                           nbinsx=max(d_hist_bins // 4, 5),
                                           marker_color="#DC2626", name="Profit < 0", opacity=0.8))
            fig_dh.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
            fig_dh.add_vline(x=np.mean(d_profits), line_dash="dot", line_color="#2563EB",
                             annotation_text=f"Mean: ${np.mean(d_profits):,.0f}",
                             annotation_position="top right")
            fig_dh.update_layout(barmode="overlay", xaxis_title="Net Profit ($)",
                                  yaxis_title="Count", height=380,
                                  legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_dh, use_container_width=True)

        with dc2:
            st.markdown("**Conditional Stats**")
            st.markdown("**When Miller wins:**")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("E[Profit | Win]",     f"${np.mean(d_won_profits):,.0f}"   if len(d_won_profits) else "N/A")
            wc2.metric("P(Profit>0 | Win)",   f"{np.mean(d_won_profits > 0):.1%}" if len(d_won_profits) else "N/A")
            wc3.metric("Trials won",          f"{len(d_won_profits):,}")
            st.markdown("**When Miller loses:**")
            lc1, lc2 = st.columns(2)
            lc1.metric("E[Loss | Lose]", f"${np.mean(d_lost_profits):,.0f}" if len(d_lost_profits) else "N/A")
            lc2.metric("Trials lost",    f"{len(d_lost_profits):,}")

            st.markdown("---")
            st.markdown("**Number of Competitors**")
            nc = pd.Series(d["n_competitors"]).value_counts().sort_index()
            fig_nc = px.bar(x=nc.index, y=nc.values / n_sim,
                            labels={"x": "Competitors", "y": "Probability"},
                            color_discrete_sequence=["#7C3AED"], text_auto=".1%")
            fig_nc.update_layout(height=230, showlegend=False)
            st.plotly_chart(fig_nc, use_container_width=True)

        # CDF + Pie chart
        dc3, dc4 = st.columns(2)
        with dc3:
            st.markdown("**Cumulative Distribution (CDF)**")
            sp = np.sort(d_profits)
            cdf = np.arange(1, len(sp) + 1) / len(sp)
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(x=sp, y=cdf, mode="lines",
                                          line=dict(color="#2563EB", width=2), name="CDF"))
            fig_cdf.add_vline(x=0, line_dash="dash", line_color="#DC2626", line_width=1,
                              annotation_text="Break-even", annotation_position="top right")
            fig_cdf.add_hline(y=1 - float(np.mean(d_profits > 0)),
                              line_dash="dot", line_color="#6B7280", line_width=1)
            fig_cdf.update_layout(xaxis_title="Net Profit ($)",
                                   yaxis_title="Cumulative Probability", height=380,
                                   legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_cdf, use_container_width=True)

        with dc4:
            st.markdown("**Outcome Breakdown**")
            n_wp = int(np.sum(d_won & (d_profits > 0)))
            n_wu = int(np.sum(d_won & (d_profits <= 0)))
            n_ls = int(np.sum(~d_won))
            fig_pie = go.Figure(go.Pie(
                labels=["Won & Profitable", "Won & Unprofitable", "Lost"],
                values=[n_wp, n_wu, n_ls],
                marker=dict(colors=["#16A34A", "#F59E0B", "#DC2626"]),
                textinfo="label+percent", hole=0.35,
            ))
            fig_pie.update_layout(height=380, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Cost breakdown
        st.markdown("---")
        st.markdown("**Cost Breakdown (winning trials only)**")
        if len(d_won_profits) > 0:
            df_dwin = pd.DataFrame({
                "Completion Cost": d["comp_costs"][d_won],
                "Bid Prep Cost":   d["prep_costs"][d_won],
                "Net Profit":      d_won_profits,
            })
            fig_db = go.Figure()
            for col, color in zip(["Completion Cost", "Bid Prep Cost", "Net Profit"],
                                   ["#F59E0B", "#3B82F6", "#10B981"]):
                fig_db.add_trace(go.Box(y=df_dwin[col], name=col, marker_color=color, boxmean=True))
            fig_db.update_layout(yaxis_title="Amount ($)", height=350)
            st.plotly_chart(fig_db, use_container_width=True)

        # Sample scenarios
        st.markdown("---")
        st.markdown("**Sample Scenarios**")
        d_n = st.slider("Number of scenarios to show", min_value=5, max_value=500,
                         value=20, step=5, key="sweep_n_scenarios")
        idx = np.arange(min(d_n, n_sim))
        d_b_bids = np.where(d["b_enters"][idx], d["bids_b_raw"][idx], np.nan)
        d_c_bids = np.where(d["c_enters"][idx], d["bids_c_raw"][idx], np.nan)
        scen_df = pd.DataFrame({
            "Scenario #":      idx + 1,
            "Bid Prep Cost":   d["prep_costs"][idx],
            "Project Cost":    d["comp_costs"][idx],
            "Comp A Bid":      d["bids_a"][idx],
            "Comp B Bid":      d_b_bids,
            "Comp C Bid":      d_c_bids,
            "Lowest Comp Bid": d["min_competitor"][idx],
            "Miller Wins?":    np.where(d["won"][idx], "✅ Yes", "❌ No"),
            "Net Profit":      d["profits"][idx],
        })
        for c in ["Bid Prep Cost", "Project Cost", "Comp A Bid", "Comp B Bid",
                  "Comp C Bid", "Lowest Comp Bid", "Net Profit"]:
            scen_df[c] = scen_df[c].apply(
                lambda v: "Did not bid" if pd.isna(v) else f"${v:,.0f}"
            )
        st.dataframe(scen_df, use_container_width=True, hide_index=True)
