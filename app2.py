import streamlit as st
st.multiselect('select a country or a set of countries',lc)
st.multiselect('select a government response or a set of government responses',lm)
st.number_input('select number of days per incidence')
st.selectbox('select a method',['hierarchical method'])
st.button('posterior predictive analysis')
