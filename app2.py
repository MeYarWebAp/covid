import streamlit as st
import base64
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy
import scipy.stats as stats
from PIL import Image, ImageDraw, ImageFont
onedrive_link ="https://1drv.ms/x/s!AquyG0uXFObDgQvfFW3Q3DW5wnNr?e=NsyagP"
@st.cache
def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl
#import skimage
#from skimage import transform as tf
plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
st.set_option('deprecation.showPyplotGlobalUse', False)
#@st.cache(allow_output_mutation=False)
def covidbook():
    crl=create_onedrive_directdownload (onedrive_link)
    #kreis=pd.read_excel(crl)
    return pd.read_excel(crl)
covidbook=covidbook()
CB=covidbook
st.dataframe(covidbook)
col1, col2,col3 = st.columns(( 1,1,1))

    

#with col2:
ec=st.selectbox("select the evaluation criteria",["","cases incidence7","cases incidence10","cases incidence14", "cases incidence7diff","cases incidence10diff","cases incidence14diff"], help="success measurement & credit assignment problem")
if ec != "":
        
    covidbook[ec].plot()
    st.pyplot() 
#with col2:   
st.dataframe(covidbook[ec].describe(percentiles=[.05,.25,.5,.75,.9,.95]).transpose())
es=st.selectbox("select the evaluation subject",["","Response Code","Response Intensity"])
if es != "":
    ax = covidbook.groupby(es)[es+' Frequency'].size().plot(kind='bar', figsize=(12,3), title='Responses type & Frequencies of Gov covid19 Responses [02.03.2020-12.06.2022]', color=colors[5])
    _ = ax.set_xlabel('Response Code')
    _ = ax.set_ylabel('Response Code Frequency')
    _ = plt.xticks(rotation=45)
    _ = ax.set_xlabel('Response code')
    _ = ax.set_ylabel('Response Code Frequency')
    _ = plt.xticks(rotation=45)
    st.pyplot()   

em=st.selectbox("select the evaluation method",["","pooled model/ bayesian analysis","partially pooled model/ hierarchical bayesian analysis"])
if em != "":
    col1, col2,col3 = st.columns(( 1,1,1))
    with col1:
        df1=st.selectbox("select the distribution function 1",["","t","LogNormal","Normal","Gamma","Beta","NegativeBinomial","Uniform","HalfNormal","Cauchy"])
        if df1 != "":
            df1p1=st.number_input("Input the parameter 1",key=11)
            df1p2=st.number_input("Input the parameter 2",key=12)   
    with col2:
        df2=st.selectbox("select the distribution function 2",["","t","LogNormal","Normal","Gamma","Beta","NegativeBinomial","Uniform","HalfNormal","Cauchy"])
        if df2 != "":
            df2p1=st.number_input("Input the parameter 1",key=21)
            df2p2=st.number_input("Input the parameter 2",key=22)
    with col3:
        df3=st.selectbox("select the distribution function 3",["","t","LogNormal","Normal","Gamma","Beta","NegativeBinomial","Uniform","HalfNormal","Cauchy"])
        if df3 != "":
            df3p1=st.number_input("Input the parameter 1",key=31)
            df3p2=st.number_input("Input the parameter 2",key=32)


