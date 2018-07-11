# Tavsiye Sistemleri (Recommender System)

Tavsiye sistemleri, veri biliminin en popüler uygulamaları arasındadır. Kullanıcının bir ögeye vereceği 'derecelendirmeyi' veya 'tercihini' tahmin etmek için kullanılır.


## Tavsiye Sistemleri Sınıfları

### Simple Recommenders

Her kullanıcı için film popülerliğine ve/veya türüne göre genelleştirilmiş önerilerde bulunur. Bu sistemin arkasındaki temel fikir daha popüler ve eleştirmenlerce beğenilen filmlerin , ortalama kitle tarafından beğenilme olasılığının yüksek olmasıdır.


Örnek olarak IMDB'den toplanan meta verileri kullanılarak basitleştirilmiş bir IMDB Top 250 film klonu oluşturulacaktır.

Adımlar:
 * Filmleri derecelendirmek için metriğe veya skora karar verilmesi
 * Her film için skorun hesaplanması
 * Filmlerin skora göre sıralanması ve en iyi sonuçların elde edilmesi.

Bu adımları gerçekleştirmeden önce film metadata veri kümesi bir pandas DataFrame'e yüklenir.

```python
#Pandas kütüphanesi eklenir
import pandas as pd

#Filmlerin Metadata verilerinin yüklenmesi
metadata = pd.read_csv('Desktop/movies_metadata.csv', low_memory=False)
```

Düşünülen en temel metriklerden biri derecelendirmedir. Ancak bu metriği kullanmanın bazı sakıncaları vardır.Birincisi filmin popülerliğini dikkate almaz. Bu nedenle 10 seçmenden 9 puan alan bir film, 10000 seçmenden 8.9 puan alan bir filmden daha iyi sayılacak. Ayrıca bu metrik son derece yüksek puanlarla daha az sayıda seçmen bulunan filmleri tercih 
etme eğiliminde olacaktır. Seçmen sayısı arttıkça Bir filmin derecelendirilmesi, filmin kalitesini yansıtan bir değere doğru düzenli olarak yaklaşır. Bir filmin kalitesini az sayıda seçmenle belirlemek zordur.

Bu eksiklikler dikkate alınarak, ortalama dereceyi ve alınan oyların sayısını dikkate alan 'ağırlıklı bir derecelendirme notu' getirmek gerekir. Böyle bir sistem 100000 seçmenden 9 puan almış bir filmin, aynı derecelendirmeye sahip ancak birkaç yüz seçmene sahip bir YouTube web serisinden daha yüksek puan almasını sağlayacaktır.

IMDB'nin Top 250 klonu oluşturulacağı için ağırlıklı derecelendirme formülü kullanılacaktır. Matematiksel olarak:

#### Weighted Rating (WR) = (v/(v+m).R)+(m/(v+m).C)


 * v->Filmin oy sayısı
 * m->Tabloda listelenmesi gereken asgari oy
 * R->Filmin ortalama derecesi
 * C->Raporun tamamındaki ortalama oylama

Verisetinde her film için V(vote_count) ve R(vote_average) değerleri zaten var. C değerini de doğrudan hesaplamak mümkündür. Belirlenmesi gereken m değeri için ise doğru bir değer yoktur. Bu değer belirli bir sayıdan az oy almış filmleri yok sayan bir filtre olarak düşünülebilir. Değeri tercihe bağlıdır.

Bu durumda 90'lık persentif kesilme olarak kullanılacaktır. Yani grafikte yer alan bir film, listedeki filmlerin en az %90'ından  daha fazla oy almalıdır.

İlk olarak, tüm filmlerin ortalama dereceleri(C) hesaplanır.

```python
#Filmlerin ortalama derecesinin (C) hesaplanması
C = metadata['vote_average'].mean()
```

m değerinin 90 persentif olarak hesaplanması için pandas kütüphanesi ve .quantile() metodu kullanılır.

```python
#En az alınabilecek oy sayısı(m) hesaplanması
m = metadata['vote_count'].quantile(0.90)
```

Ardından oylama sayılarına bağlı olarak grafik için uygun filmler filtrelenir.

```python
#Filmlerin Filtrelenmesi
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
```

.copy() ile yeni q_movies DataFrame'i oluşturulup filtrelenen filmler bu DataFrame'e aktarıldı ki orjinal metadata verileri etkilenmesin.

```python
#Filmlerin Filtrelenmesi
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
```

Şimdi her film için metriğin hesaplanması gerekiyor. Bunun için weight_rating() fonksiyonu tanımlanıp, filmler için yeni bir özellik (score) tanımlanacaktır. 

```python
#Metriği hesaplayan fonksiyon
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

#score değeri oluşturulur. weighted_rating() fonksiyonu ile değeri hesaplanır ve atanır.
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

```

Son olarak, DataFrame'in score özelliğie göre sıralanması ve en iyi 15 filmin başlığının, oy sayısının, oy oranının ve ağırlıklı puanının gösterilmesi.

```python
#Filmlerin sıralanması
q_movies = q_movies.sort_values('score', ascending=False)

#En iyi 15 filmin yazdırılması
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
```

Programın Çalıştırılması:

![SimpleRecommenders](https://github.com/merveyapnaz/Tavsiye-Sistemleri-Recommender-System/blob/master/Screenshots/SimpleRecommenders.PNG)

### Content Based Recommender

#### Plot Description Based Recommender

Bu bölümde belirli filme benzer filmleri öneren bir sistem oluşturulmaya çalışılacaktır. Daha spesifik olarak tüm filmlerin açıklamalarına göre çift yönlü benzerlik puanı hesaplanacak ve bu benzerlik puanına göre filmler önerilecektir.

Açıklama için verikümesindeki 'overview' özelliği kullanılabilir. 

Verilerin mevcut halleri ile iki açıklama arasındaki benzerliği hesaplamak mümkün değildir. Bunun için her bir 'overview' ın ya da belgenin kelime vektörlerini hesaplamak gerekir. 

Her belge için  Vadeli Frekans-Ters Belge Sıklığı/Term Frequency-Inverse Document Frequency (TF-IDF) vektörleri hesaplanacaktır. Bu, her bir sütunun genel kelime haznesinde bir kelimeyi temsil ettiği ve her sütunun bir filmi temsil ettiği bir matris verecektir.

Özünde  TF/IDF puanı, bir belgede meydana gelen bir sözcüğün sıklığı olup, içerdiği belge sıklığıyla azalır. Bu, genel olarak gözden geçirmelerde sıkça ortaya çıkan kelimelerin önemini azaltmak için yapılır.

Neyse ki scikit-learn'ün birkaç satırda TF-IDF matrisi üreten yerleşik bir sınıfı vardır : TfIdVectorizer

NOT: Normalde bu kısımda elimizdeki orjinal metadata verisetini kullanmamız gerekiyor fakat kişisel bilgisayarlarımızın çoğunluğu yeterli donanıma sahip olmadığı için çoğunlukla bellek hatası almaktayız. Ben kendi yaptığım örnekte daha önceden oluşturmuş olduğumuz ve sadece belirli bir sayıdan fazla oy almış filmleri içeren q_movies Dataframe'ini kullandım.

```python
#scikit-learn'den TfIdfVectorizer'ın yüklenmesi
from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF nesnesinin tanımlanıp 'the','a' gibi İngilizce kelimelerin kaldırılması
tfidf = TfidfVectorizer(stop_words='english')

#Nan değerlerinin boşluk ile değiştirilmesi
q_movies['overview'] = q_movies['overview'].fillna(' ')

#Verileri yerleştirip dönüştürerek gerekli TF-IDF matrisinin oluşturulması
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])
```

Artık elde edilen bu matris ile bir benzerlik puanı hesaplanabilir. Bunun için birkaç aday var; Öklid, Pearson ve Kosinüs benzerlik puanları gibi. Hangi puanın en iyi olduğuna dair doğru bir cevap yoktur. Farklı puanlar farklı senaryolarda iyi çalışır. Bu noktada farklı denemeler yapmak iyi fikirdir. 

Burda farklı iki film arasındaki benzerliği gösteren sayısal bir miktar hesaplamak için kosinüs benzerliği kullanılacaktır(büyüklükten bağımsız ve nispeden daha hızlı ve basit olduğu için). Matematiksel olarak şöyledir:


#### cosine(x,y)=(x.y⊺)/(||x||.||y||)


TF-IDF vectorizer'ı kullanıldığı için, ürünün hesaplanması doğrudan kosinüs benzerlik puanı verecektir. Bunun için daha hızlı olması açısından cosine_similarities() yerine  sklearn linear_kernel() kullanılacaktır.

```python
#linear_kernel'in yüklenmesi
from sklearn.metrics.pairwise import linear_kernel

#Kosinüs benzerlik matrisinin hesaplanması
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

```

Bir filmin başlığını giriş olarak alan ve en benzer 10 filmi listeleyen bir fonksiyon tanımlanacaktır. Öncelikle bunun için film başlıklarının ve DataFrame endekslerinin bir tersine haritalanması gerekir. Başka bir deyişle metadata DataFrame'indeki bir filmin dizinini tanımlamak için bir başlık gerekir. 

```python
#İndekslerin ve film başlıklarının ters haritasını oluşturulması
indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()
```


Şimdi program tavsiye işlevini tanımlamak için iyi bir konumdadır. İzlenecek adımlar şunlardır:

* Başlığı verilen filmin dizininin alınması.
* Tüm filmler için, o film için kosinüs benzerlik puanlarının listesinin alınması. 
* İlk elemanın pozisyonu ve ikincinin benzerlik puanının olduğu bir tupl listesine dönüştürülmesi
* Benzerlik puanlarına dayanılarak bu tupl listesinin sıralanması; Yani ikinci element
* Bu listenin en iyi 10 ögesinin elde edilmesi. İlk öge görmezden gelinmeli çünkü bir filme en çok benzeyen filmin kendisidir.
* Elde edilen ögelerin dizinlerine karşılık gelen başlıkalrın döndürülmesi.

```python
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

```
Programın Çalıştırılması:

![PlotDescriptionBasedRecommender](https://github.com/merveyapnaz/Tavsiye-Sistemleri-Recommender-System/blob/master/Screenshots/PlotDescriptionBasedRecommender.PNG)

Sistem benzer açıklamalara sahip filmleri bulmakta işe yarasa da tavsiyelerin kalitesi o kadar da iyi durmuyor.

#### Credits, Genres and Keywords Based Recommender

Daha iyi meta verilerinin kullanılmasıyla önerilerin kalitesi de artacaktır. Şimdi de şu verilere dayanılarak bir danışman oluşturulacaktır: 3 en iyi oyuncu, yönetmen, ilgili tür ve film arşivi.

Anahtar kelimeler, oyuncu ve ekip verileri, mevcut veri kümesinde mevcut değildir. Bu nedenle, ilk adım, bunları ana DataFrame'me yüklemek ve birleştirmek olacaktır.


```python
# Anahtar kelimelerin ve kredilerin yüklenmesi
credits = pd.read_csv('Desktop/credits.csv')
keywords = pd.read_csv('Desktop/keywords.csv')

#Id lerin int e çevrilmesi. Birleştirme için gerekli.
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
q_movies['id'] = q_movies['id'].astype('int')

#Anahtar kelimelerin ve kredilerin ana metadata çerçevesi ile birleştirilmesi
q_movies = q_movies.merge(credits, on='id')
q_movies = q_movies.merge(keywords, on='id')
```

Yeni özelliklerden, oyunculardan ve anahtar kelimelerden, en önemli üç aktörü, yönetmeni ve bu filmle ilişkili anahtar kelimeleri çıkarmak gerekir. Şu anda veriler sınırlı listeler halinde mevcut. Bunları kullanılabilecek forma dönüştürmek gerekir.


```python
#Belirtilen özellikler ile ilgili phyton nesnelerinin ayrılması
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(literal_eval)

```

Ardından, her özellikten gerekli bilgileri almaya yardımcı olacak fonksiyonlar yazılabilir. Öncelikle NaN sabitine erişmek için NumPy paketi yüklenir .get_director() fonksiyonu yazımında kullanılacaktır.

```python
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

```

Sıradaki işlem, isimleri ve anahtar kelimeleri küçük harfe dönüştürmek ve aralarındaki tüm boşlukları silmek olacaktır. Bu, vektörleştiricinin "Johnny Depp" ve "Johnny Galecki" nin Johnny'sini aynı şekilde saymaması için yapılır.

```python
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

```

Şimdi de sistem "metadata çorbası" yaratabilecek bir konumda. Bu, vectorizer'a beslemek istenen tüm meta verileri (yani, aktörler, yönetmenler ve anahtar kelimeler) içeren bir dizedir.

```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Yeni bir çorba özelliğinin oluşturulması
q_movies['soup'] = q_movies.apply(create_soup, axis=1)

```

Sonraki adımlar Plot Description Based Recommender sistemi ile aynıdır. Önemli bir fark ise, CountVectorizer()TF-IDF yerine kullanmaktır. Bunun sebebi, bir aktörün / yönetmenin varlığını, göreceli olarak daha fazla filmde oynamış veya yönetmiş olması halinde aşağı çekmek istememektir.

```python
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

```

Artık, ikinci argüman olarak cosine_sim2 matrisini göndererek get_recommendations() fonksiyonu yeniden kullanılabilir.

![CreditsGenresAndKeywordsBasedRecommender](https://github.com/merveyapnaz/Tavsiye-Sistemleri-Recommender-System/blob/master/Screenshots/CreditsGenresAndKeywordsBasedRecommender.PNG)



Kullanılan Veriseti : [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)

Yararlanılan Kaynak : [Recommender Systems in Python: Beginner Tutorial](https://www.datacamp.com/community/tutorials/recommender-systems-python)
