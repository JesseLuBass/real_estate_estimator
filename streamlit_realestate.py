from matplotlib.figure import Figure
from numpy.lib.function_base import select
from matplotlib.backends.backend_agg import RendererAgg
from PIL import Image
from streamlit_lottie import st_lottie
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor

import os
import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import requests
import urllib.request
import pickle



# Setting plotting style
matplotlib.use("agg")

_lock = RendererAgg.lock

plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Page configuration
st.set_page_config(layout="wide")

# get lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_wRN8db.json')
st_lottie(lottie_book, speed=1, height=200, key="initial")



# Setting the title
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Vancouver Housing Estimate')

with row0_2:
    st.write('')

row0_2.subheader(
    'A Web App by [Jesse Lu](https://github.com/JesseLuBass)')

row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))


# Short description 
with row1_1:
    st.markdown("Hey there! Welcome to Jesse's Housing Price Estimate App. This app reads in a DataFrame of housing information and returns a listing price predicition for each observation.")
    st.write('You can also input your own custom info for a prediction.')
    st.markdown(" 👇 **Please select which subset of the Vancouver housing information DataFrame you would like the model to predict on.** 👇")


row2_spacer1, row2_1, row2_spacer2 = st.beta_columns((.1, 3.2, .1))

df = pd.read_csv('df_short_test.csv').drop(['Price'], axis=1)
first_20 = df.iloc[:20,:]
first_100 = df.iloc[:1000,:]
entire_df = df.copy()


with row2_1:    
    
    df_dict = {'20 properties':first_20, '1000 properties': first_100, '5000 properties': entire_df}
    options = list(df_dict.keys())
    selected = st.selectbox('Select a subset of the DataFrame', options = options)
    chosen_df = df_dict[selected]
    st.dataframe(chosen_df)


row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))


with row3_1, _lock:
    st.subheader('What kind of properties are they?')
    plot_df = chosen_df['Type'].dropna()
    fig, ax = plt.subplots()
    ax.hist(plot_df)
    st.pyplot(fig)

with row3_2:
    st.subheader('How long have they been on the market for?')
    
    plot_df = chosen_df[chosen_df['Days On Market'] < 100]['Days On Market']
    fig, ax = plt.subplots()
    ax.hist(plot_df)
    plt.xlabel('Days On Market')
    plt.ylabel('Number of Properties')
    st.pyplot(fig)

st.write('')
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))

with row4_1:
    st.subheader('Do they have a scenic view?')
    plot_df = chosen_df['View']
    
    fig, ax = plt.subplots()
    
    ax.pie(plot_df.value_counts(), labels= ('Yes','No'),autopct='%1.1f%%')
    st.pyplot(fig)

with row4_2:
    st.subheader('Where are they? (5 most common)')
    index = chosen_df['Area'].value_counts()[:5].index.tolist()
    
    plot_df = chosen_df[chosen_df['Area'].isin(index)]['Area']
    
    fig, ax = plt.subplots()
    ax.hist(plot_df)
    st.pyplot(fig)

st.subheader('Here are the predictions')
model = pickle.load(open('data_3_model_2','rb'))
res = model.predict(chosen_df)
chosen_df['Prediction'] = res
chosen_df = chosen_df[['Prediction','Type', 'S/A','Area', 'Yr Blt','Age','Days On Market',
       'Total Bedrooms', 'Floor Area -Grand Total',
       'View', 'No. Floor Levels']]
st.write(chosen_df)



st.header('You can make your own custom data from the selections below here')

sa_option = sorted(list(df['S/A'].unique()))
typedwel_option = list(df.TypeDwel.unique())
type_option = list(df.Type.unique())
area_option = list(df.Area.unique())

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row4_1:
    city = st.selectbox('Which city is this house situated in?', area_option)
with row4_2:
    sa = st.selectbox('Option for Property Area', sa_option)

row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row5_1:
    type= st.selectbox('Option for the type of property', type_option)
with row5_2:
    typedwel = st.selectbox('Option for type of dwelling', typedwel_option)

row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row6_1:
    view = st.selectbox('Does the property have a view?', ('Yes', 'No'))
with row6_2:
    year = st.slider('What year was the property built?', 1980,2021,2000)

row7_space1, row7_1, row7_space2, row7_2, row7_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row7_1:
    age = st.slider('The age of the property', 6.,40.,20.)
with row7_2:
    num_rooms = st.slider('How many rooms are there in the property?', 5.,20., 10., step=1.)

row8_space1, row8_1, row8_space2, row8_2, row8_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row8_1:
    bedrooms = st.slider('How many bedrooms are there on the property?', 0.,10.,3.)
with row8_2:
    bathrooms = st.slider('How many bathrooms are there on the property?',0.,7.,3.)

row9_space1, row9_1, row9_space2, row9_2, row9_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row9_1:
    area = st.slider('The floor area for this property in square feet',400.,3000.,1700., step=1.)
with row9_2:
    lotsize =st.slider('The lot size for this property in square feet',0., 10000., 3000.,step=1. )

row10_space1, row10_1, row10_space2, row10_2, row10_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row10_1:
    dom = st.slider('How many days have the property been on the market?', 2,60,30)




if st.button('Make a prediction on my selection above'):
    data = {
        'Prop Type':'Residential Attached', 
        'S/A':sa, 
        'Yr Blt':year, 
        'TypeDwel':typedwel, 
        '# of Kitchens':1,
        '# Rms':num_rooms, 
        'Age':age, 
        'Floor Area -Grand Total':area, 
        'Area': city, 
        'Type': type,
        'Total Bedrooms':bedrooms, 
        'Total Baths':bathrooms, 
        'Days On Market':dom, 
        'List Date':737752,
        'Lot Sz (Sq.Ft.)':3307, 
        'View':view, 
        'Fireplaces':1., 
        'Parking Places - Total':2.,
        'No. Floor Levels':2., 
        'Dist to School/School Bus changed':'Close',
        'Distance to Pub/Rapid Tr changed':'Close', 
        'Zoning changed':'CD',
        'Realtor remarks sentiment classified':'positive',
        'Public remarks sentiment classified':'positive',
        
        }
    custom_df =pd.DataFrame(data, index=[0])

try:
    st.write(int(model.predict(custom_df)))
except: 
    pass