import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime as dt

lw=st.number_input('Lower Window',-10,0,-1)
uw=st.number_input('Upper Window',0,10,1)
mode=st.selectbox('Select Seasonality',['additive','multiplicative'])
periods=st.number_input('Forecast Period',0,365,30)

events1=st.text_input('Event 1 Name')
# events2=st.text_input('Event 2 Name')
# events3=st.text_input('Event 3 Name')
# events4=st.text_input('Event 4 Name')
# events5=st.text_input('Event 5 Name')

eventd1=st.date_input(label='Event 1 Date Range: ',value=(dt(year=1990, month=1, day=1, hour=00, minute=00), 
                    dt(year=2023, month=1, day=30, hour=00, minute=00)),)
# eventd2=st.date_input(label='Event 2 Date Range: ',value=(dt(year=1990, month=1, day=1, hour=00, minute=00), 
#                     dt(year=2023, month=1, day=30, hour=00, minute=00)),)
# eventd3=st.date_input(label='Event 3 Date Range: ',value=(dt(year=1990, month=1, day=1, hour=00, minute=00), 
#                     dt(year=2023, month=1, day=30, hour=00, minute=00)),)
# eventd4=st.date_input(label='Event 4 Date Range: ',value=(dt(year=1990, month=1, day=1, hour=00, minute=00), 
#                     dt(year=2023, month=1, day=30, hour=00, minute=00)),)
# eventd5=st.date_input(label='Event 5 Date Range: ',value=(dt(year=1990, month=1, day=1, hour=00, minute=00), 
#                     dt(year=2023, month=1, day=30, hour=00, minute=00)),)

try:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    @st.experimental_memo
    def read_file(uploaded_file):
        data2=pd.read_csv(uploaded_file)
        return data2

    df=read_file(uploaded_file)
    df.columns=[df.columns[0].replace(df.columns[0],'ds'),df.columns[1].replace(df.columns[1],'y')]
    freq=st.selectbox('Choose Frequency',["D","M","Y","s","min","H"])
    rmp=st.radio('Run Model',['n','y'])

    if rmp=='y':
        set_random_seed(40)
        m = NeuralProphet(n_changepoints=150,daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True,num_hidden_layers=3,loss_func='MAE')
        event1 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1)})
        # event2 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1)})
        # event3 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1)})
        # event4 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1)})
        # event5 = pd.DataFrame({'event': events1,'ds': pd.to_datetime(eventd1)})

        # enames=[events1,events2,events3,events4,events5]
        # events_df = pd.concat((event1,event2,event3,event4,event5))
        enames=[events1]
        events_df = pd.concat([event1])

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
