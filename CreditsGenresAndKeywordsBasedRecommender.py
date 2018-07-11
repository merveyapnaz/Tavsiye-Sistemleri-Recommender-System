#Pandas kütüphanesi eklenir
import pandas as pd

#Filmlerin Metadata verilerinin yüklenmesi
metadata = pd.read_csv('Desktop/Tavsiye-Sistemleri-Recommender-System/DataSets/movies_metadata.csv', low_memory=False)

#En az alınabilecek oy sayısı(m) hesaplanması
m = metadata['vote_count'].quantile(0.90)

#Filmlerin Filtrelenmesi
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

#scikit-learn'den TfIdfVectorizer'ın yüklenmesi
from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF nesnesinin tanımlanıp 'the','a' gibi İngilizce kelimelerin kaldırılması
tfidf = TfidfVectorizer(stop_words='english')

#Nan değerlerinin boşluk ile değiştirilmesi
q_movies['overview'] = q_movies['overview'].fillna(' ')

#Verileri yerleştirip dönüştürerek gerekli TF-IDF matrisinin oluşturulması
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

#linear_kernel'in yüklenmesi
from sklearn.metrics.pairwise import linear_kernel

#Kosinüs benzerlik matrisinin hesaplanması
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#İndekslerin ve film başlıklarının ters haritasını oluşturulması
indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()

#Film başlığını giriş olarak alan ve en çok benzeyen filmleri çağıran fonksiyon
def get_recommendations(title, cosine_sim=cosine_sim):
    
    #Film başlığı ile eşleşen dizinin alınması
    idx = indices[title]

    #Bu film ile tüm filmlerin çift yönlü benzerlik puanlarının alınması
    sim_scores = list(enumerate(cosine_sim[idx]))

    #Filmlerin benzerlik puanlarına göre sıralanması
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #En benzer 10 filmin puanlarının alınması
    sim_scores = sim_scores[1:11]

    #Film indekslerinin alınması
    movie_indices = [i[0] for i in sim_scores]

    #En yakın 10 en iyi filmin döndürülmesi
    return q_movies['title'].iloc[movie_indices]

# Anahtar kelimelerin ve kredilerin yüklenmesi
credits = pd.read_csv('Desktop/Tavsiye-Sistemleri-Recommender-System/DataSets/credits.csv')
keywords = pd.read_csv('Desktop/Tavsiye-Sistemleri-Recommender-System/DataSets/keywords.csv')

#Hatalı Idli satırların kaldırılması
#metadata = metadata.drop([19730, 29503, 35587])

#Id lerin int e çevrilmesi. Birleştirme için gerekli.
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
q_movies['id'] = q_movies['id'].astype('int')

#Anahtar kelimelerin ve kredilerin ana metadata çerçevesi ile birleştirilmesi
q_movies = q_movies.merge(credits, on='id')
q_movies = q_movies.merge(keywords, on='id')

#Belirtilen özellikler ile ilgili phyton nesnelerinin ayrılması
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(literal_eval)

#NumPy 'nin yüklenmesi
import numpy as np

#Yönetmenin adının ekip özelliğinden alınması. Eğer yoksa Nan döndür.
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

#Listenin en üst 3 elementini döndürür ya da tüm listeyi; hangisi daha fazla ise
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        
        # 3'ten fazla elemanın mevcut olup olmadığını kontrol eder. Varsa, sadece ilk üçünü geri döndürür. Yoksa, tüm listeyi döndür.
        if len(names) > 3:
            names = names[:3]
        return names

    # Eksik / hatalı biçimlendirilmiş verilerde boş liste döndür
    return []

#Uygun bir formda olan yeni yönetmen, oyuncu, tür ve anahtar kelime özelliklerinin tanımlanması
q_movies['director'] = q_movies['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(get_list)

#Tüm stringleri küçük harfe dönüştürme ve aralarındaki boşlukları kaldırma
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Direktör var mı kontrol edin. Değilse boş dizge döndürün
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#clean_Data fonksiyonunun özelliklere uygulanması
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    q_movies[feature] = q_movies[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Yeni bir çorba özelliğinin oluşturulması
q_movies['soup'] = q_movies.apply(create_soup, axis=1)  

# CountVectorizer'ın içe aktarılması ve sayı matrisinin oluşturulması
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(q_movies['soup'])

#Sayma matrisine göre Kosinüs Benzerlik matrisinin hesaplanması
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#Ana DataFrame'in dizininin sıfırlanıp ve ters haritalamanın oluşturulması.
q_movies = q_movies.reset_index()
indices = pd.Series(q_movies.index, index=q_movies['title'])

while(True):
    film=input("Filmin Adı Nedir?(Programdan çıkmak için \'Bitir\' yazınız)")
    if(film == 'Bitir'):
        break
    else:
        print(get_recommendations(film, cosine_sim2))