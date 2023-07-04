import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv("CAR.csv")
st.title("Car Price Predictor")

nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])

if nav == "Home":
    st.image("car pic.jpg", width=600)
    if st.checkbox("Show Table"):
        password = st.number_input("Enter Authentication Key")
        if password == 1234.0:
            st.table(df)


    graph = st.selectbox("What kind of Graph ? ", ["Wheelbase-Price", "Boreratio-Price","Curbweight-Price",
                                                   "Enginesize-Price","Horsepower-Price"])
    if graph == "Wheelbase-Price":
        plt.figure(figsize=(10, 5))
        fig=sns.scatterplot(x=df['wheelbase'], y=df['price'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if graph == "Boreratio-Price":
        plt.figure(figsize=(10, 5))
        fig=sns.scatterplot(x=df['boreratio'], y=df['price'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if graph == "Curbweight-Price":
        plt.figure(figsize=(10, 5))
        fig=sns.scatterplot(x=df['curbweight'], y=df['price'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if graph == "Enginesize-Price":
        plt.figure(figsize=(10, 5))
        fig=sns.scatterplot(x=df['enginesize'], y=df['price'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if graph == "Horsepower-Price":
        plt.figure(figsize=(10, 5))
        fig=sns.scatterplot(x=df['horsepower'], y=df['price'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    bar = st.selectbox("What kind of Bar Graph ? ", ["Fueltype-Price", "Aspiration-Price", "Carbody-Price",
                                                     "Enginetype-Price", "Drivewheel-Price"])

    if bar == "Fueltype-Price":
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df['fueltype'], y=df['price'])
        plt.xticks(rotation='vertical')
        st.pyplot()
    if bar == "Aspiration-Price":
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df['aspiration'], y=df['price'])
        plt.xticks(rotation='vertical')
        st.pyplot()
    if bar == "Carbody-Price":
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df['carbody'], y=df['price'])
        plt.xticks(rotation='vertical')
        st.pyplot()
    if bar == "Enginetype-Price":
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df['enginetype'], y=df['price'])
        plt.xticks(rotation='vertical')
        st.pyplot()
    if bar == "Drivewheel-Price":
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df['drivewheel'], y=df['price'])
        plt.xticks(rotation='vertical')
        st.pyplot()

    st.image("aj.jpg", width=150)

if nav == "Prediction":
    st.subheader("Prediction")
    st.image("car1 pic.jpg", width=200)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['fueltype'] = le.fit_transform(df['fueltype'])
    df['aspiration'] = le.fit_transform(df['aspiration'])
    df['carbody'] = le.fit_transform(df['carbody'])
    df['drivewheel'] = le.fit_transform(df['drivewheel'])
    df['enginetype'] = le.fit_transform(df['enginetype'])
    y = df['price']
    x = df.drop(columns=['price'])
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=70, random_state=5, max_samples=0.9,
                                  max_features=0.2, max_depth=15)
    model.fit(x_train, y_train)

    ftype = st.selectbox('FuelType(gas-1   diesel-0)', df['fueltype'].unique())

    # aspiration
    asp = st.selectbox('Aspiration(std-0  turbo-1)', df['aspiration'].unique())

    # carbody
    cbody = st.selectbox('CarBody(convertible-0  hardtop-1  wagon-4  hatchback-2  sedan-3)', df['carbody'].unique())

    # drivewheel
    wheel = st.selectbox('DriveWheel(rwd-2  fwd-1  4wd-0)', df['drivewheel'].unique())

    # enginetype
    etype = st.selectbox('EngineType(ohc-2  ohcf-3  dohc-0  ohcv-4  I-1  rotor-5)', df['enginetype'].unique())

    # stroke
    stroke = st.selectbox('Stroke', [2, 4])

    # wheelbase
    base = st.number_input('WheelBase of the Car(inches)')

    # curbweight
    weight = st.number_input('CurbWeight of the Car(pounds)')

    # lenght
    length = st.number_input('Lenght of the Car(cm)')

    # width
    width = st.number_input('Width of the Car(foot)')

    # height
    height = st.number_input('Height of the Car(foot)')

    # enginesize
    size = st.number_input('EngineSize of the Car')

    # boreratio
    ratio = st.number_input('BoreRatio of the Car')

    # horsepower
    power = st.number_input('HorsePower of the Car(hp)')

    # citympg
    cmpg = st.number_input('Citympg of the Car(km/l)')

    # highwaympg
    hmpg = st.number_input('Highwaympg of the Car(km/l)')

    if st.button('Predict Price'):
        query = np.array([ftype, asp, cbody, wheel,etype,stroke,base, weight,length,width,height, size, ratio, power,
                          cmpg,hmpg])

        query = query.reshape(1, 16)
        st.title(
            "The predicted price of this configuration of car model: Rs " + str(int((model.predict(query)[0]) * 79.65)))
        

    st.image("aj.jpg", width=150)

if nav == "Contribute":
    st.subheader("Contribute to the dataset")
    ftype = st.selectbox('FuelType', df['fueltype'].unique())

    # aspiration
    asp = st.selectbox('Aspiration', df['aspiration'].unique())

    # carbody
    cbody = st.selectbox('CarBody', df['carbody'].unique())

    # drivewheel
    wheel = st.selectbox('DriveWheel', df['drivewheel'].unique())

    # enginetype
    etype = st.selectbox('EngineType', df['enginetype'].unique())

    # stroke
    stoke = st.selectbox('Stroke', [2, 4])

    # wheelbase
    base = st.number_input('WheelBase of the Car(inches)')

    # curbweight
    weight = st.number_input('CurbWeight of the Car(pounds)')

    # lenght
    length = st.number_input('Lenght of the Car(cm)')

    # width
    width = st.number_input('Width of the Car(foot)')

    # height
    height = st.number_input('Height of the Car(foot)')

    # enginesize
    size = st.number_input('EngineSize of the Car')

    # boreratio
    ratio = st.number_input('BoreRatio of the Car')

    # horsepower
    power = st.number_input('HorsePower of the Car(hp)')

    # citympg
    cmpg = st.number_input('Citympg of the Car(km/l)')

    # highwaympg
    hmpg = st.number_input('Highwaympg of the Car(km/l)')

    prc = st.number_input('Price of the Car')

    password = st.number_input("Enter Authentication Key")

    if st.button("submit"):
        if password == 1234.0:
            to_add = {"fueltype":[ftype],"aspiration":[asp],"carbody":[cbody],"drivewheel":[wheel],"enginetype":[etype],
                        "stroke":[stoke],"wheelbase":[base],"curbweight":[weight],"carlength":[length],"carwidth":[width],
                        "carheight":[height],"enginesize":[size],"boreratio":[ratio],"horsepower":[power],"citympg":[cmpg],
                        "highwaympg":[hmpg],"price":[prc]}
            to_add = pd.DataFrame(to_add)
            to_add.to_csv("CAR.csv", mode='a', header=False, index=False)
            st.success("Submitted")
        else:
            st.title("Cheater")
    st.image("aj.jpg",width=150)