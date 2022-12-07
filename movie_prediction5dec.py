import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()


genre_list = [
    "Action and Thriller",
    "Adventure",
    "Animation",
    "Comedy",
    "Comedy and Drama",
    "Comedy and Crime",
    "Comedy and Romance",
    "Crime",
    "Drama",
    "Drama and Crime",
    "Drama and History",
    "Drama and Romance",
    "Drama and Thriller",
    "Family",
    "Horror",
    "Horror and thriller",
    "Romance",
    "Thriller",
    "others"
]

date_list = [data['release_date']]



#release_date = 25

def show_prediction_page():
    #st.write(data)
    st.title("Previsão de Bilheteria de Filme")

    st.write("""### Insira as informações do seu filme abaixo: """)
    is_collection = st.radio("Seu filme faz parte de uma franquia/coleção?", ("Sim", "Não"))
    budget = st.slider("Qual o seu orçamento?\n(em milhões de dólares)", 0, 900, 450)
    genre = st.selectbox("Qual o gênero do seu filme?", genre_list)
    popularity = st.slider("Qual a popularidade?", 0, 9000, 4500)
    release_date = st.number_input('Ano de lançamento:', 1900, 2020)
    runtime = st.slider("Qual a duração?\n(em minutos)", 30, 240, 120)
    vote = st.slider("Qual a média dos votos da crítica?", 0, 100, 5)

    ok = st.button("Fazer previsão")
    if ok:
        if is_collection == "Sim":
            is_collection = 1
        else:
            is_collection = 0

        from sklearn.preprocessing import LabelEncoder
        le_revenue = LabelEncoder()
        
        genre_list_transformed = le_revenue.fit_transform(genre_list)
        for i in range(len(genre_list)):
            if genre_list[i] == genre:
                genre = genre_list_transformed[i]
        #release_date = le_revenue.fit_transform(['release_date'])
        # date_list_transformed = le_revenue.fit_transform(date_list)
        # for i in range(len(date_list)):
        #     release_date = date_list_transformed[i]
        #date_list = [data['release_date']]

        
        y = data['revenue']
        X = data.drop('revenue', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
        from sklearn.neural_network import MLPClassifier
        X = X_train
        y = y_train
        clf= MLPClassifier(hidden_layer_sizes=(150,200,250), max_iter=300,activation = 'relu',solver='lbfgs',random_state=1,alpha=1e-5)
        clf.fit(X, y)
        MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='lbfgs',random_state=1,alpha=1e-5)
        
        X = np.array([[is_collection, budget, genre, popularity, release_date, runtime, vote]])
        revenue = clf.predict(X)

        if revenue[0] == 0:
            revenue = "Entre 100 e 500 mil"
        elif revenue[0] == 1:
            revenue = "Entre 100 e 300 milhões"
        elif revenue[0] == 2:
            revenue = "Entre 10 e 20 milhões"
        elif revenue[0] == 3:
            revenue = "Entre 2 e 5 milhões"
        elif revenue[0] == 4:
            revenue = "Entre 20 e 50 milhões"
        elif revenue[0] == 5:
            revenue = "Entre 300 e 500 milhões"
        elif revenue[0] == 6:
            revenue = "Entre 5 e 10 milhões"
        elif revenue[0] == 7:
            revenue = "Entre 50 e 80 milhões"
        elif revenue[0] == 8:
            revenue = "Entre 500 e 900 mil"
        elif revenue[0] == 9:
            revenue = "Entre 80 e 100 milhões"
        elif revenue[0] == 10:
            revenue = "Entre 900 mil e 2 milhões"
        elif revenue[0] == 11:
            revenue = "Acima de 500 milhões"

        #st.write(f"X --> {X}")
        st.subheader("Sua bilheteria esperada:")
        st.subheader(f"{revenue}")






show_prediction_page()
