import streamlit as st

from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'OpenAI'], 
        icons=['houses', 'gear'], menu_icon="cast", default_index=1)

if selected == 'Home':
    st.title('ML Exhaust')
elif selected == 'OpenAI':
    st.title('Exploration of OpenAI awesomeness')
