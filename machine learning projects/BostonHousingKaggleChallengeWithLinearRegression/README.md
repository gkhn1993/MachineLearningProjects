## Bostondaki Ev fiyatlarını tahmin etmeye çalışan program
### Proje öngereksinimleri:
* Numpy, Pandas, Scikit-learn, Matplotlib, Jupyter Notebook
### Proje Konusu:
* Bu projede sklearn kütüphanesi içerisinde yerleşik olarak gelen veri setlerinde load_boston isimli veri setini kullandık. 
* Bu veri kümesinde toplamda 506 tane kayıt ve 13 tane özelliğimiz var. Bu özellikler ise şu şekildedir:
![gokhan](https://user-images.githubusercontent.com/47734102/132104392-29dcb659-8e99-4794-9766-9e028e77ee94.png)
- Veri kümemize ev fiyatlarını tahmin etmek için price adında yeni bir sütun ekledim.
![price](https://user-images.githubusercontent.com/47734102/132104490-48ad51f0-75d4-459f-b870-9c06c21a200f.png)
- Veri setime ait bilgiyi info metodu ile, veri setimde yer alan özelliklerin standart sapması, ortalaması, maksimum ve minimum değerlerini almak içinse describe() metodunu kullandım.
![describe](https://user-images.githubusercontent.com/47734102/132104550-1111dae1-ae2e-44d6-ada5-c9b427f764ed.png)
![info](https://user-images.githubusercontent.com/47734102/132104558-79180c33-2291-48d5-b91f-b055252aa702.png)
* Veri kümemde yer alan değerleri eğitim ve test verisi olmak üzere iki kategoriye böldüm. Veri kümemin yüzde 20'sini test, yüzde 80'ini eğitim için kullandım.
* Bu durumda 102 kayıt test için geriye kalan 404 kayıt ise eğitim verisi olarak kullanıldı. 
### Proje Amacı: 
<p> Kısaca özetlemek gerekirse sklearn kütüphanesi içinde hazır olarak bana gelen veri kümelerinden load_boston veri setini kullanarak
 boston şehrindeki ev fiyatlarını 13 özelliğe bağlı olarak gerçek hayattaki değerlerine yakın olacak şekilde tahmin etmesi için makine öğrenmesi algoritmalarından sadece biri 
 olan linear regression yani doğrusal regresyon algoritmasını kullandık. Kısaca doğrusal regresyon analizi, bir değişkenin değerini başka bir değişkenin değerine göre tahmin etmek için kullanılır.
 Programın sonunda gerçek değerler ve tahmin edilen değerler arasındaki korelasyonu matplotlib kütüphanesinin içinde yer alan
 çizgi grafiğiyle gösterdik.</p>

### Proje Çıktısı:
![output](https://user-images.githubusercontent.com/47734102/132104878-b2182825-6e30-47d6-b0a0-3573547f6c9e.png)

# Doğrusal Regresyonun Sonuçları:
` from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(ytest, y_pred)
  print("Mean Square Error : ", mse)`
![mean](https://user-images.githubusercontent.com/47734102/132105071-ff916e51-c999-4269-9d4b-06a6e5c3575c.png)

* Bu fonksiyon matematiksel olarak aşağıdaki resimde gösterildiği gibi hesaplanır.

![formul](https://user-images.githubusercontent.com/47734102/132105102-483c868b-d411-4b29-9294-9acecca59a37.jpg)
 
### Proje Sonucunu Yorumlama:
<p> Kurmuş olduğumuz modelimizin fiyatları ne kadar büyük bir doğrulukla tahmin ettiğini görmek için mean_squared_error() fonksiyonunu kullandık
ve sonuç olarak bu fonksiyon bize 33.45 değerini döndürdü. Bu şu anlama geliyor. Bizim kurduğumuz model ev fiyatlarını tahmin etmekte sadece yüzde 66 oranında doğru sonuç veriyor
ki bu çok iyi bir oran değil. Bu veri kümesi üzerinde başka makine öğrenmesi algoritmaları veya teknikleri uygulayarak tahmin değerlerini artırmak ve metrik hatasını olabildiğince
azaltarak ev fiyatlarını gerçek hayattaki değerlerine olabildiğince yakın bir şekilde tahmin edebilirim. Bu model ev fiyatlarını diğer 13 özelliğe bağımlı olarak tahmin etmekte
 sadece yüzde 66 oranında doğru sonuç vermektedir.</p>
