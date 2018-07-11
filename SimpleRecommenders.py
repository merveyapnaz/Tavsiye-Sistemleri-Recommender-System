#Pandas kütüphanesi eklenir
import pandas as pd

#Filmlerin Metadata verilerinin yüklenmesi
metadata = pd.read_csv('Desktop/Tavsiye-Sistemleri-Recommender-System-/DataSets/movies_metadata.csv', low_memory=False)

#Filmlerin ortalama derecesinin (C) hesaplanması
C = metadata['vote_average'].mean()

#En az alınabilecek oy sayısı(m) hesaplanması
m = metadata['vote_count'].quantile(0.90)

#Filmlerin Filtrelenmesi
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

#Metriği hesaplayan fonksiyon
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

#score değeri oluşturulur. weighted_rating() fonksiyonu ile değeri hesaplanır ve atanır.
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Filmlerin sıralanması
q_movies = q_movies.sort_values('score', ascending=False)

#En iyi 15 filmin yazdırılması
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))