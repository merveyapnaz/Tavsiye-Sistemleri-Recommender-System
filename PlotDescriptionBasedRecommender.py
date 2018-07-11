#Pandas kütüphanesi eklenir
import pandas as pd

#Filmlerin Metadata verilerinin yüklenmesi
metadata = pd.read_csv('Desktop/Tavsiye-Sistemleri-Recommender-System-/DataSets/movies_metadata.csv', low_memory=False)

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


while(True):
    film=input("Filmin Adı Nedir?(Programdan çıkmak için \'Bitir\' yazınız)")
    if(film == 'Bitir'):
        break
    else:
        print(get_recommendations(film))
