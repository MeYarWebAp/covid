"""
Streamlit app for Bayesian worst-case analysis of a new feature rollout.
Based on this article
https://www.crosstab.io/articles/confidence-interval-interpretation, the scenario is
that we're doing a staged rollout of a new feature on our company website, and we need
to decide if we should proceed to the next stage. In this scenario, we want a
*non-inferiority* analysis, not a traditional A/B test superiority analysis. We also
want to use a Bayesian approach, so that our conclusions about the true click-rate have
the interpretation that matches most decision-makers' intuition.
The Streamlit app code specifically is also discussed in this article:
https://www.crosstab.io/articles/streamlit-review.
NOTE
- Figures are saved in SVG format if this script is run in a REPL instead of Streamlit.
  It would be better to save directly to PNG, but I can't figure out how to control the
  resolution of the Altair figure saver. This way, I can use Inkscape to convert to PNG
  with the desired resolution.
"""

import base64
import altair as alt
import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions

onedrive_link ="https://1drv.ms/x/s!AquyG0uXFObDgQE0adb-zEMfPAI7?e=I2dgYa"
@st.cache
def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


alt.renderers.set_embed_options(scaleFactor=2)


## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.markdown("<h2 style='text-align: center;color:blue'>COVID-19 Response-Measures Future-Efficiency Estimator</h2>", unsafe_allow_html=True )
crl=create_onedrive_directdownload (onedrive_link)
dataset=pd.read_excel(crl)
dataset.drop(dataset.columns[[0]], axis = 1, inplace = True)

st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.columns(3)

tick_size = 12
axis_title_size = 16


## Simulate data and the distribution domain


## User inputs on the control panel
st.sidebar.subheader("Prior belief about the Response-Measures efficiency rate")
if st.sidebar.checkbox("display dataset?"):
    st.sidebar.dataframe(dataset)
prior_daily_cases = st.sidebar.number_input(
    "Number of prior days",
    min_value=1,
    max_value=None,
    value=100,
    step=1,
    help="The higher this number, the stronger your prior belief about Response-Measures efficiency rate before observing updated data.",
)
number_of_implemented_responses= st.sidebar.number_input(
    "Number of observed days",
    min_value=1,
    max_value=None,
    value=10,
    step=1,
    help="Number of days implemented measures are observed",
)


st.sidebar.subheader("Decision criteria")
worst_case_threshold = st.sidebar.slider(
    "Worst-case deaths cases ratio threshold",
    min_value=0.001,
    max_value=0.2,
    value=0.005,
    step=0.001,
    help="A deaths cases ratio below this value is defined to be the worst-case scenario",
)

worst_case_max_proba = st.sidebar.slider(
    "Max acceptable worst-case probability",
    min_value=0.001,
    max_value=0.2,
    value=0.005,
    step=0.001,
    help="The larger this threshold, the more risk we're willing to accept that the worst-case scenario might happen.",
)
#dataset.drop(dataset.columns[[0]], axis = 1, inplace = True)
df_1 = dataset.iloc[:prior_daily_cases,:]
df_2 = dataset.iloc[prior_daily_cases:(prior_daily_cases + number_of_implemented_responses),:]
df_2=df_2.reset_index()
data=df_2
#st.dataframe(data)
## Define the prior
sm = df_1["daily deaths cases ratio"].values
f = Fitter(sm,distributions=["beta"])
f.fit()

prior = stats.beta(f.fitted_param["beta"][0], f.fitted_param["beta"][1])


## Show key results over time. The index value indicates the data for that day has been
# observed.
results = pd.DataFrame(
    {
        "mean": [prior.mean()],
        "p10": [prior.ppf(0.1)],
        "p90": [prior.ppf(0.9)],
    },
    index=[-1],
)

posterior = copy.copy(prior)
assert id(posterior) != id(prior)

for t in range(number_of_implemented_responses):

    # This is the key posterior update logic, from the beta-binomial conjugate family.
    posterior = stats.beta(
        posterior.args[0] + data.loc[t, "daily deaths"],
        posterior.args[1] + data.loc[t, "daily not deaths"],
    )

    results.at[t] = {
        "mean": posterior.mean(),
        "p10": posterior.ppf(0.1),
        "p90": posterior.ppf(0.9),
    }

graphicmax=0.99999
## Get the max useful daily deaths cases ratio value to show in the distribution plots
xmax = max(prior.ppf(graphicmax), posterior.ppf(graphicmax))
distro_grid = np.linspace(0, xmax, 300)


## Draw the prior
prior_pdf = pd.DataFrame(
    {"daily deaths cases ratio": distro_grid, "prior_pdf": [prior.pdf(x) for x in distro_grid]}
)

fig = (
    alt.Chart(prior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("daily deaths cases ratio", title="daily deaths cases ratio"),
        y=alt.Y("prior_pdf", title="Probability density"),
        tooltip=[
            alt.Tooltip("daily deaths cases ratio", title="daily deaths cases ratio", format=".3f"),
            alt.Tooltip("prior_pdf", title="Probability density", format=".3f"),
        ],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(x="x", tooltip=[alt.Tooltip("x", title="Worst-case threshold")])
)

worst_case_prior_pdf = prior_pdf[prior_pdf["daily deaths cases ratio"] < worst_case_threshold]

worst_case_area = (
    alt.Chart(worst_case_prior_pdf)
    .mark_area(opacity=0.5)
    .encode(x="daily deaths cases ratio", y="prior_pdf")
)

fig = alt.layer(fig, threshold_rule, worst_case_area).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_prior.svg")

left_col.subheader("Prior belief about the daily deaths cases ratio")
left_col.altair_chart(fig, use_container_width=True)


## Draw the final posterior
posterior_pdf = pd.DataFrame(
    {
        "daily deaths cases ratio": distro_grid,
        "posterior_pdf": [posterior.pdf(x) for x in distro_grid],
    }
)

fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("daily deaths cases ratio", title="daily deaths cases ratio", scale=alt.Scale(domain=[0, 0.99999])),
        y=alt.Y("posterior_pdf", title="Probability density"),
        tooltip=[
            alt.Tooltip("daily deaths cases ratio", title="daily deaths cases ratio", format=".3f"),
            alt.Tooltip("posterior_pdf", title="Probability density", format=".3f"),
        ],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(x="x", tooltip=[alt.Tooltip("x", title="Worst-case threshold")])
)

fig = alt.layer(fig, threshold_rule).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

left_col.subheader("Posterior belief about the daily deaths cases ratio")
left_col.altair_chart(fig, use_container_width=True)


## Draw the data
base = alt.Chart(data).encode(alt.X("day", title="Experiment day"))

volume_fig = base.mark_bar(color="#ffbb78", size=12).encode(
    y=alt.Y(
        "daily cases", axis=alt.Axis(title="Number of daily cases", titleColor="#ff7f0e")
    ),
    tooltip=[alt.Tooltip("daily cases", title="Num daily cases")],
)

rate_fig = base.mark_line(size=4).encode(
    y=alt.Y("daily deaths cases ratio", axis=alt.Axis(title="daily deaths cases ratio", titleColor="#1f77b4")),
    tooltip=[alt.Tooltip("daily deaths cases ratio", title="daily deaths cases ratio", format=".3f")],
)

fig = (
    alt.layer(volume_fig, rate_fig)
    .resolve_scale(y="independent")
    .configure_axis(labelFontSize=tick_size, titleFontSize=axis_title_size)
)
fig= data["daily deaths cases ratio"].plot()


if st.util.env_util.is_repl():
    fig.save("worst_case_data.svg")
annotation_layer = (
    alt.Chart(data)
    .mark_line()
    .encode(
        x="date",
        y="daily deaths cases ratio"
    )
    .interactive()
)
middle_col.subheader("Observed data")
middle_col.altair_chart((annotation_layer).interactive(), use_container_width=True)


## Draw the posterior zoomed in
xmin = posterior.ppf(0.0001)
xmax = posterior.ppf(0.9999)
distro_grid = np.linspace(xmin, xmax, 300)

posterior_pdf = pd.DataFrame(
    {
        "daily deaths cases ratio": distro_grid,
        "posterior_pdf": [posterior.pdf(x) for x in distro_grid],
    }
)

distro_fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("daily deaths cases ratio", title="daily deaths cases ratio", scale=alt.Scale(domain=[xmin, xmax])),
        y=alt.Y("posterior_pdf", title="Probability density"),
        tooltip=[
            alt.Tooltip("daily deaths cases ratio", title="daily deaths cases ratio", format=".3f"),
            alt.Tooltip("posterior_pdf", title="Probability density", format=".3f"),
        ],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red", clip=True)
    .encode(x="x", tooltip=[alt.Tooltip("x", title="Worst-case threshold")])
)

worst_case_post_pdf = posterior_pdf[posterior_pdf["daily deaths cases ratio"] < worst_case_threshold]

worst_case_area = (
    alt.Chart(worst_case_post_pdf)
    .mark_area(opacity=0.5)
    .encode(x="daily deaths cases ratio", y="posterior_pdf")
)

fig = alt.layer(distro_fig, threshold_rule, worst_case_area).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_posterior.svg")

middle_col.subheader("Zoomed-in posterior belief")
middle_col.altair_chart(fig, use_container_width=True)


## Draw key results over time
results.reset_index(inplace=True)
out = results.melt(id_vars=["index"])

ts_mean = (
    alt.Chart(results)
    .mark_line()
    .encode(
        x="index",
        y="mean",
    )
)

band = (
    alt.Chart(results)
    .mark_area(opacity=0.5)
    .encode(
        x=alt.X("index", title="Experiment day"),
        y=alt.Y("p10", title="daily deaths cases ratio"),
        y2="p90",
        tooltip=[
            alt.Tooltip("index", title="Experiment day"),
            alt.Tooltip("p10", title="80% credible interval lower", format=".3f"),
            alt.Tooltip("p90", title="80% credible interval upper", format=".3f"),
        ],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"y": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(y="y")
)

fig = alt.layer(ts_mean, band, threshold_rule).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_posterior_ts.svg")

right_col.subheader("Posterior over time")
right_col.altair_chart(fig, use_container_width=True)


## Write key outputs in the control panel
right_col.subheader("Results and decision")

observed_daily_cases = data["daily cases"].sum()
observed_daily_deaths = data["daily deaths"].sum()
observed_daily_deaths_cases_ratio = observed_daily_deaths / observed_daily_cases
worst_case_proba = posterior.cdf(worst_case_threshold)

if worst_case_proba < worst_case_max_proba:
    decision = "GO"
    emoji = "white_check_mark"
else:
    decision = "NO GO"
    emoji = "no_entry_sign"

right_col.markdown(f"**Observed daily cases:** {observed_daily_cases:,}")
right_col.markdown(f"**Observed daily deaths cases ratio:** {observed_daily_deaths_cases_ratio:.4f}")
right_col.markdown(f"**Mean posterior daily deaths cases ratio:** {posterior.mean():.4f}")
right_col.markdown(
    f"**80% credible region for daily deaths cases ratio:** [{posterior.ppf(0.1):.4f}, {posterior.ppf(0.9):.4f}]"
)
right_col.markdown(
    f"**P(daily deaths cases ratio < than critical threshold):** {worst_case_proba:.2%}"
)
right_col.subheader(f"***Final decision:*** {decision} :{emoji}:")
