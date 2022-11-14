import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime as dt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.title('Time Series Forecasting with Neural Prophet')

try:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    @st.experimental_memo
    def read_file(uploaded_file):
        data2=pd.read_csv(uploaded_file)
        return data2

    df=read_file(uploaded_file)

    with st.expander("Select Date & Observed Value Columns",expanded=True):
        c1, c2 = st.columns((1, 1))
        x=c1.selectbox('Select Date Column',df.columns)
        y=c2.selectbox('Select Observed Value Column',df.columns)

    with st.expander("Select Event Names & their Dates",expanded=True):
        c3, c4 = st.columns((1, 1))
        events1=c3.text_input('Event 1 Name')
        eventd1=c3.date_input(label='Event 1 Date Range: ',value=(dt(year=1990, month=1, day=1), 
                            dt(year=2023, month=1, day=30)),)
        events2=c4.text_input('Event 2 Name')
        eventd2=c4.date_input(label='Event 2 Date Range: ',value=(dt(year=1990, month=1, day=1), 
                            dt(year=2023, month=1, day=30)),)

    with st.expander("Select the Lower & Upper Window for the Events",expanded=True):
        c5, c6, c7 = st.columns((1, 1, 1))
        lw=c5.number_input('Lower Window',-10,0,-1)
        uw=c6.number_input('Upper Window',0,10,1)
        mode=c7.selectbox('Select Seasonality',['Additive','Multiplicative'])

    with st.expander("Choose the Forecast Period with its Frequency",expanded=True):
        c8, c9 = st.columns((1, 1))
        periods=c8.number_input('Forecast Period',0,365,30)
        freq=c9.selectbox('Choose Frequency',["D","M","Y","s","min","H"])

    rmp=st.radio('Run Model',['n','y'])

    df['ds'],df['y']=df[x],df[y]
    df=df[['ds','y']]


    if rmp=='y':
        set_random_seed(40)
        m = NeuralProphet(n_changepoints=150,daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True,num_hidden_layers=1,loss_func='MAE')
        event1 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1).date})
        event2 = pd.DataFrame({'event': events2,'ds': pd.to_datetime(eventd2).date})
        # event1 = pd.DataFrame({'event': events1,'ds': eventd1})
        # event2 = pd.DataFrame({'event': events2,'ds': eventd2})
        enames=[events1,events2]
        events_df = pd.concat([event1,event2])

        events_df=events_df[events_df['event']!='']
        for i in range(len(enames)):
            if enames[i]!='':
                m=m.add_events([enames[i]],lower_window=lw,upper_window=uw,mode=mode)
        history_df = m.create_df_with_events(df, events_df)
        metrics = m.fit(history_df, freq=freq)
        future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=periods)
        forecast = m.predict(df=future) 
        fig = m.plot(forecast)
        fig_comp = m.plot_components(forecast)
        fig_param = m.plot_parameters()
        st.pyplot(fig)
        st.pyplot(fig_comp)
        st.pyplot(fig_param)
        st.dataframe(forecast)

        def convert_df(df):
            return df.to_csv().encode('utf-8')

        forecast_df = convert_df(forecast)
        
        try:
            if forecast_df is not None:
                st.download_button(label="Download data as CSV",data=forecast_df,file_name='NeuralProphet_with_events_results.csv',mime='text/csv',)
        except:
            st.write('Choose Something')
except:
    st.write('Choose Something')
