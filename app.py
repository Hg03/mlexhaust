import streamlit as st
from markdownlit import mdlit
import plotly_express as px
import pandas as pd
import openai
from streamlit_option_menu import option_menu
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import joblib
from sklearn import model_selection, preprocessing, svm, impute, metrics
import numpy as np
import time

st.set_page_config(layout='wide',page_title='mlexhaust')

@st.cache_data
def load_data():
    data = pd.read_csv('data/Life-Expectancy-Data.csv')
    return data

def generate_img(prompt):
    response = openai.Image.create(
    prompt=prompt,n=1,size="256x256")
    image_url = response['data'][0]['url']
    return image_url

def home():
    mdlit('# ML [red]Exhaust[/red] 👾')
    one, two = st.columns(2)
    mdlit("### Hi 👋, Welcome to All ML enthusiasts, My name is [green] Harish Gehlot [/green] and presently I am Data Science Intern at @([violet] Katonic.ai [/violet])(https://katonic.ai). Here In this streamlit app, I am going to implement lots of projects with [blue] basic stuff [/blue] as well a [red] complicated stuff [/red]")
    mdlit('#### So what is MLExhaust ??')
    mdlit("> It's kinda Upgrading my mind [blue]time[/blue] to [blue]time[/blue] by exploring the awesomeness of usecases of [orange]Machine Learning[/orange]")
    #st.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/harish-gehlot-5338a021a/)",unsafe_allow_html=True)
    #st.markdown("[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AvratanuBiswas/PubLit)",unsafe_allow_html=True)
    #st.subheader('Links')
    #mdlit(" <br> 1. @(LinkedIn)(https://www.linkedin.com/in/harish-gehlot-5338a021a/) <br> 2. @(Github)(https://github.com/Hg03) <br> 3. @(Blog)(https://mlpapers.substack.com/)")


def openai_():
    mdlit('## [green] OpenAI [/green] exploration 🔥')
    user_api_key = None
    submit_btn = None
    if 'generated' not in st.session_state:
        st.session_state.generated = []
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'input' not in st.session_state:
        st.session_state.input = ''
    if 'stored_session' not in st.session_state:
        st.session_state.stored_session = []
    with st.expander("If you have your own OpenAI API key, or mine will exhausted"):
        user_api_key = st.text_input(label = 'Your API key',placeholder='Your API key',type='password')
        submit_btn = st.button('Submit')

    
    if not submit_btn:
        openai.api_key = st.secrets['OPENAI_API_KEY']
        mdlit("> ### Let's [red]first[/red] look at the Dalle API to generate some awesome images")
        input_text_for_image = st.text_input("Put your imagination here to generate image")
        if input_text_for_image:
            if st.button('Generate Image'):
                url_img = generate_img(input_text_for_image)
                st.info(f"View your awesome image at {url_img}")
        mdlit("> ### Let's [blue]Secondly[/blue] look at ChatGPT api through which we can start a conversation")
        input_text = st.text_input("You: ","Hi, What you need to talk about?", key="input")
        llm = OpenAI(temperature=0.7,model_name="text-davinci-003")

        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10)
        
        CONVERSATION = ConversationChain(llm=llm,prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,memory=st.session_state.entity_memory)
        if input_text:
            output = CONVERSATION.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

    else:
        openai.api_key = user_api_key
        mdlit("> ### Let's [red]first[/red] look at the Dalle API to generate some awesome images")
        input_text_for_image = st.text_input("Put your imagination here to generate image")
        if input_text_for_image:
            if st.button('Generate Image'):
                url_img = generate_img(input_text_for_image)
                st.info(f"View your awesome image at {url_img}")
        mdlit("> ### Let's [blue]Secondly[/blue] look at ChatGPT api through which we can start a conversation")
        input_text = st.text_input("You: ","Hi, What you need to talk about?", key="input")
        llm = OpenAI(temperature=0.7,openai_api_key=user_api_key,model_name="text-davinci-003")

        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10)
        
        CONVERSATION = ConversationChain(llm=llm,prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,memory=st.session_state.entity_memory)
        if input_text:
            output = CONVERSATION.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


    
#     openai.api_key = st.secrets['OPENAI_API_KEY']



def life_expectancy():
    mdlit("## Life Expectancy [blue]Prediction[/blue]")
    dashboard, prediction = st.tabs(['Little Analysis Dashboard','Prediction'])
    df = load_data()
    with dashboard:
        st.plotly_chart(px.pie(df,names='Year',values='Infant_deaths',title='Infant Deaths per year'))
        st.plotly_chart(px.histogram(df,x='Country',y='Life_expectancy',color='Region',title='Life expectancy per country & region'))
        st.plotly_chart(px.scatter(df,x='Adult_mortality',y='Alcohol_consumption',color='Year',title='Effect of alcohol consumption on adult mortality'))
    with prediction:
        mdlit("Predict the [green]expectancy[/green] of life based on following values")
        inputs = {'Country':'','Infant_deaths':np.nan,'Under_five_deaths':np.nan,'Adult_mortality':np.nan,'Alcohol_consumption':np.nan,'Hepatitis_B':np.nan,'Measles':np.nan,'BMI':np.nan,'Polio':np.nan,'Diphtheria':np.nan,'Incidents_HIV':np.nan,'GDP_per_capita':np.nan,'Population':np.nan,'Thinness_ten_nineteen_years':np.nan,'Thinness_five_nine_years':np.nan,'Schooling':np.nan,'Economy_status_Developed':np.nan,'Economy_status_Developing':np.nan}
        with st.form('prediction form'):
            inputs['Country'] = st.selectbox('Select your Country',df.Country.unique().tolist())
            inputs['Infant_deaths'] = st.slider('Amount of Infant Deaths',min_value=1.0,max_value=140.0)
            inputs['Under_five_deaths'] = st.slider('Amount of Deaths under 5',min_value=2.0,max_value=225.0)
            inputs['Adult_mortality'] = st.number_input('Adult Mortality Rate',1.0,1000.0)
            inputs['Alcohol_consumption'] = st.slider('Alcohol Consumption',0.0,30.0)
            with st.container():
                col1, col2 = st.columns(2)
                inputs['Hepatitis_B'] = col1.number_input('Hepatitis B',0,100)
                inputs['Measles'] = col2.number_input('Measles',0,100)
            inputs['BMI'] = st.slider('BMI',min_value=10.0,max_value=40.0)
            with st.container():
                col1, col2 = st.columns(2)
                inputs['Polio'] = col1.number_input('Polio',min_value=0.0,max_value=100.0)
                inputs['Diphtheria'] = col2.number_input('Diphtheria',min_value=1,max_value=100)
            inputs['Incidents_HIV'] = st.slider('Incidenys HIV',0.1,30.0)
            with st.container():
                col1, col2 = st.columns(2)
                inputs['GDP_per_capita'] = col1.number_input('GDP per capita',130,120000)
                inputs['Population_mln'] = col2.number_input('Population ',0.0,1400.0)
            inputs['Thinness_ten_nineteen_years'] = st.slider('Thinness Ten Nineteen Years',0.1,30.0)
            inputs['Thinness_five_nine_years'] = st.slider('Thinness Five Nine Years',0.1,30.0)
            with st.container():
                col1, col2, col3 = st.columns(3)
                inputs['Schooling'] = col1.number_input('Schooling',1.0,20.0)
                inputs['Economy_status_Developed'] = col2.number_input('Economy Status Developed',0,1)
                inputs['Economy_status_Developing'] = col3.number_input('Economy Status Developing',0,1)

            submit = st.form_submit_button('Submit')
            if submit:
                model = joblib.load('models/life_expectancy_model.joblib')
                input_frame = pd.DataFrame(np.array(list(inputs.values())).reshape(1,19),columns = inputs.keys())
                prediction = model.predict(input_frame)
                my_bar = st.progress(0, text='Rate of Life Expectancy Leads to')
                for percent_complete in range(100):
                    time.sleep(0.1)
                    if percent_complete <= round(prediction[0]):
                        my_bar.progress(percent_complete + 1,text='Rate of Life Expectancy Leads to')
                st.write(f'🎉 {prediction[0]} 🎉')


                #st.info(f'Rate of Life Expectancy according to the model is {prediction[0]}')
                #st.write(input_frame)
            	    




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
	              visibility: hidden;
                    }
            footer:after {
              content:'Made by Harish Gehlot'; 
              visibility: visible;
              display: block;
              position: relative;
              #background-color: purple;
              padding: 5px;
              top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    selected = st.selectbox("📝 Main Menu", ["🏠 Home", '🔥 OpenAI','🛟 Life Expectancy Prediction'])
    mdlit("## **[yellow]Social Links[/yellow]** ")
    st.markdown("[🌐 linkedIn](https://www.linkedin.com/in/harish-gehlot-5338a021a/)")
    mdlit("[🐱 Github](https://github.com/Hg03/mlexhaust)")

if selected == '🏠 Home':
    home()
elif selected == '🔥 OpenAI':
    openai_()
elif selected == '🛟 Life Expectancy Prediction':
    life_expectancy()
	
	
