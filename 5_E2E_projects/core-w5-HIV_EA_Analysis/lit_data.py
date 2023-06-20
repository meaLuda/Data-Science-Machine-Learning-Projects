# Data visualization with StreamLit

# By : ELiud Munyala

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt



data = pd.read_csv(r'/home/eliud_luda/Desktop/Moringa_Prep/W5_prep/final_data.csv')

df = pd.DataFrame(data)

"""
# INVESTIGATING EFFECTIVENESS OF PEER EDUCATION ON HIV_&_AIDS AMONG ADOLESCENTS IN EAST_AFRICA
"""

"""

## Our data works on the following premises:
1. Age groups 10-19
2. Years 2010 to 2020
3. Countries(Kenya,Uganda,Tanzania)
4. Residence(Rural or Urban)


"""



chart_visual = st.sidebar.selectbox('Select Charts/Plot type', ('Pie Chart','Box plot'))

selected_status = st.sidebar.selectbox('Select Status on HIV by Country to ', options=['Total_Living','Total_Mortality','Percent_of_Educated','Percent_in_Rural_Urban'])

"""
### Select Box Plot for totals and Pie chart to compare countries with different values
"""

if chart_visual == 'Pie Chart':
    if selected_status == 'Total_Living':
        fig = px.pie(df, values=df['Total_Living'], names=df['Country'], title='Total Confirmed People living With HIV IN East Africa')
        st.plotly_chart(fig)
    if selected_status == 'Total_Mortality':
        fig_1 = px.pie(df, values=df['Total_Mortality'], names=df['Country'], title='Total HIV Mortality in East Africa')
        st.plotly_chart(fig_1)
    if selected_status == 'Percent_of_Educated':
      fig_2 = px.pie(df, values=df['Percent_of_Educated'], names=df['Country'], title='HIV Education level East Africa')
      st.plotly_chart(fig_2)
    if selected_status == 'Percent_in_Rural_Urban': 
        fig_3 = px.pie(df, values=df['Percent_in_Rural_Urban'], names=df['Country'], title='HIV Education level East Africa')
        st.plotly_chart(fig_3)
elif chart_visual == 'Box plot':
    fig_4 = px.bar(df, x="Country", y=["Total_Living","Total_Mortality","Total_of_new_Infect","Total"],title="Total for People livind with HIV, and Mortality", barmode='group', height=400)
    st.plotly_chart(fig_4)

    fig_5 = px.bar(df, x="Country", y=["%_living_educated","%_mortality_educated","%_new_educated","%_new_infected_educated"],title="Total percentage of People educated on and living with HIV, and Mortality", barmode='group', height=400)
    st.plotly_chart(fig_5)