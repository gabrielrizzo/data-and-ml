from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import re

def preprocess_text(text_list):
  new_text_list = []
  for word in text_list:
      word = re.sub(r'[^a-zA-Z\s]', '', word)
      word = word.lower()
      word = ' '.join(word.split())
      new_text_list.append(word)
  return new_text_list


def train_knn(mainAxis, subAxis):
  clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
  clf.fit(mainAxis, subAxis)
  return clf

def train_nb(mainAxis, subAxis):
  nb = MultinomialNB(alpha=0.1)
  nb.fit(mainAxis, subAxis)
  return nb

def predict(clf, text, model_name):
  result = clf.predict(text)
  print('O Resultado é: ', result[0], 'no modelo: ', model_name)

def calculate_accuracy(clf, X_test, y_test, model_name):
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print('A acurácia no conjunto de teste é: ', accuracy, 'no modelo: ', model_name)

def init_models():
  textos = [
  "O novo lançamento da Apple",
  "Nvidia lança um novo chip de GPU",
  "Resultado do jogo de ontem",
  "Eleições presidenciais",
  "Atualização no mundo da tecnologia",
  "Campeonato de futebol",
  "Real madrid ganhou a champions league",
  "Política internacional",
  "Deputado aprova lei de reforma tributária",
  "Novo iPhone com câmera revolucionária",
  "GPUs 9070XT se saem melhor que as concorrentes da Nvidia",
  "Botafogo vence de goleada o Corinthians",
  "Deputado aprova lei que ajuda a população a se manter saudável",
  "Fluminense vence o Palmeiras",
  "Lula vence bolsonaro nas eleições 2022",
  "Bolsonaro vence Haddad nas eleições 2018",
  "Eleições",
  ]
  categorias = ["tecnologia", "tecnologia", "esportes", "política", "tecnologia", "esportes", "esportes", "política", "política", "tecnologia", "tecnologia", "esportes", "política", "esportes", "política", "política", "política"]

  # Convertendo textos em uma matriz de contagens de tokens
  vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
  X = vectorizer.fit_transform(preprocess_text(textos))

  # Dividindo os dados em conjuntos de treinamento e teste
  X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.1, random_state=42)

  knn_X_train, knn_X_test, knn_y_train, knn_y_test = train_test_split(X, categorias, test_size=0.1, random_state=42)

  # Treinando o classificador
  # clf = MultinomialNB()
  # clf = KNeighborsClassifier(n_neighbors=3)
  # clf.fit(X_train, y_train)

  knn_model = train_knn(knn_X_train, knn_y_train)
  nb_model = train_nb(X_train, y_train)
  # Shaping the data
  text_to_predict = ["Nokia lança novo celular com câmera de 108MP"]
  x_to_predict = vectorizer.transform(text_to_predict)

  # predict(knn_model, x_to_predict, 'KNN')
  # predict(nb_model, x_to_predict, 'NB')

  # calculate_accuracy(knn_model, knn_X_test, knn_y_test, 'KNN')
  # calculate_accuracy(nb_model, X_test, y_test, 'NB')

  return vectorizer, nb_model, knn_model

def use_nb_model(text_to_predict):
    vectorizer, nb_model, knn_model= init_models()
    x_to_predict = vectorizer.transform([text_to_predict])
    predict(nb_model, x_to_predict, 'NB')

# Predição e Avaliação
# y_pred = clf.predict(X_test)

# print(f"Acurácia: {accuracy_score(y_test, y_pred)}")