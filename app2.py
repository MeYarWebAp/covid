"""

"""
import itertools
import base64
import altair as alt
import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn import preprocessing
#st.stop()
#import seaborn.apionly as sns
#%matplotlib inline
plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00','#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
st.set_option('deprecation.showPyplotGlobalUse', False)
onedrive_link ="https://1drv.ms/x/s!AquyG0uXFObDgQXeo9qIu_prTFHx?e=1Behqf"
onedrive_link_h ="https://1drv.ms/x/s!AquyG0uXFObDgQrnt4QZ_YHZk1uB?e=FTqMBY"
@st.cache
def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


alt.renderers.set_embed_options(scaleFactor=2)


## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called

crl_h=create_onedrive_directdownload (onedrive_link_h)
dataset_h=pd.read_excel(crl_h)
dataset_h.drop(dataset_h.columns[[0,1]], axis = 1, inplace = True)
#dataset_h.drop(dataset_h.columns[[0,1]], axis = 1, inplace = True)
st.write(dataset_h)
covidbook=dataset_h

ax = covidbook.groupby('Response Type')['Response Intensity'].size().plot(
    kind='bar', figsize=(12,3), title='Number of implementation per response', color=colors[0])
_ = ax.set_xlabel('Previous Sender')
_ = ax.set_ylabel('Number of messages')
_ = plt.xticks(rotation=45)
_ = ax.set_xlabel('Response code')
_ = ax.set_ylabel('Number of implementation')
_ = plt.xticks(rotation=45)
st.pyplot()
indiv_traces = {}

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
responses_idx = le.fit_transform(covidbook['Response Type'])
responses = le.classes_
n_responses = len(responses)
if 1==2:
    for p in responses:
        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=0, upper=100)
            mu = pm.Uniform('mu', lower=0, upper=100)

            data = covidbook[covidbook['Response Type']==p]['Normalized_daily_deaths'].values
            y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=data)

            y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)

            start = pm.find_MAP()
            step = pm.Metropolis()
            trace = pm.sample(20000, step, start=start, progressbar=True)

            indiv_traces[p] = trace

    fig, axs = plt.subplots(3,2, figsize=(12, 6))
    axs = axs.ravel()
    y_left_max = 2
    y_right_max = 2000
    x_lim = 60
    ix = [3,4,6]

    for i, j, p in zip([0,1,2], [0,2,4], responses[ix]):
        axs[j].set_title('Observed: %s' % p)
        axs[j].hist(covidbook[covidbook['Response Type']==p]['Normalized_daily_deaths'].values, range=[0, x_lim], bins=x_lim, histtype='stepfilled')
        axs[j].set_ylim([0, y_left_max])

    for i, j, p in zip([0,1,2], [1,3,5], responses[ix]):
        axs[j].set_title('Posterior predictive distribution: %s' % p)
        axs[j].hist(indiv_traces[p].get_values('y_pred'), range=[0, x_lim], bins=x_lim, histtype='stepfilled', color=colors[1])
        axs[j].set_ylim([0, y_right_max])

    axs[4].set_xlabel('Normalized_daily_deaths')
    axs[5].set_xlabel('Normalized_daily_deaths')

    plt.tight_layout()
    st.pyplot()

    combined_y_pred = np.concatenate([v.get_values('y_pred') for k, v in indiv_traces.items()])

    x_lim = 100
    y_pred = trace.get_values('y_pred')

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(211)

    fig.add_subplot(211)

    _ = plt.hist(combined_y_pred, range=[0, x_lim], bins=x_lim, histtype='stepfilled', color=colors[1])   
    _ = plt.xlim(1, x_lim)
    _ = plt.ylim(0, 20000)
    _ = plt.ylabel('Frequency')
    _ = plt.title('Posterior predictive distribution')

    fig.add_subplot(212)

    _ = plt.hist(covidbook['Normalized_daily_deaths'].values, range=[0, x_lim], bins=x_lim, histtype='stepfilled')
    _ = plt.xlim(0, x_lim)
    _ = plt.xlabel('Normalized_daily_deaths')
    _ = plt.ylim(0, 20)
    _ = plt.ylabel('Frequency')
    _ = plt.title('Distribution of observed data')
    st.pyplot()



with pm.Model() as model:
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=50)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=10)
    
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=50)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=50)
    
    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_responses)
    mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_responses)
    
    y_est = pm.NegativeBinomial('y_est', 
                                mu=mu[responses_idx], 
                                alpha=alpha[responses_idx], 
                                observed=covidbook['Normalized_daily_deaths'].values)
    
    y_pred = pm.NegativeBinomial('y_pred', 
                                 mu=mu[responses_idx], 
                                 alpha=alpha[responses_idx],
                                 shape=covidbook['Response Type'].shape)
    
    start = pm.find_MAP()
    step = pm.Metropolis()
    hierarchical_trace = pm.sample(20000, step, progressbar=True)
_ = pm.plot_trace(hierarchical_trace)#[12000:], varnames=['mu','alpha','hyper_mu_mu', 'hyper_mu_sd','hyper_alpha_mu','hyper_alpha_sd'])
st.pyplot()








