from sklearn.naive_bayes import GaussianNB

with open("sinav_veri_seti_son.txt", "r", encoding="utf-8") as dosya:
    duygu_liste = dosya.readlines()
    dosya.close()

gorusler = []
etiketler = []  

for satir in duygu_liste:
    bolunmus_satir = satir.split(";;")
    gorusler.append(bolunmus_satir[0])
    etiketler.append(bolunmus_satir[1])


OlumluKelimeler = ["kesinlikle alın", "harika", "muhteşem", "muhtesem", "müthiş", "muthis", "çok güzel", "cok guzel", "teşekkürler", "tesekkurler", "teşekkür", "tesekkur",
                   "tavsiye ederim", "alınabilir", "alinabilir", "beğendim", "begendim", "sorunsuz", "başarılı", "basarili", "süper", "super", "yeterli", "makul"]

OlumsuzKelimeler = ["sakın almayın", "aşırı kötü", "asiri kotu", "memnun degilim", "tavsiye etmem", "bozuldu", "beğenmedim", "begenmedim", "iade", "iyi değil", "iyi degil", "memnun değilim",
                    "memnun etmedi", "çalışmıyor", "calismiyor", "kırık", "kirik", "çöp", "cop", "pişman oldum", "pisman oldum", "vasat", "perişan", "perisan", "çok kötü", "cok kotu"]

X = []
y = []

for index in range(len(gorusler)):
    mesaj = gorusler[index].lower()
    x_vector = [0] * len(OlumluKelimeler)
    for gorus in mesaj.split(" "):
        if gorus in OlumluKelimeler:
            x_vector[OlumluKelimeler.index(gorus)] = 1
        elif gorus in OlumsuzKelimeler:
            x_vector[OlumsuzKelimeler.index(gorus)] = 2

    X.append(x_vector)

    if etiketler[index] == "Olumlu\n":
        y.append(0)
    elif etiketler[index] == "Olumsuz\n":
        y.append(1)
    else:
        y.append(2)

model = GaussianNB()
model.fit(X, y)

yeni_gorus = "Muhtesem bir ürün, tavsiye ederim."
yeni_x_vector = [0] * len(OlumluKelimeler)
for gorus in yeni_gorus.split(" "):
    if gorus in OlumluKelimeler:
        yeni_x_vector[OlumluKelimeler.index(gorus)] = 1
    if gorus in OlumsuzKelimeler:
        yeni_x_vector[OlumsuzKelimeler.index(gorus)] = 2

tahmin = model.predict([yeni_x_vector])

if tahmin[0] == 0:
    print("Olumlu")
elif tahmin[0] == 1:
    print("Olumsuz")
else:
    print("Tarafsız")

X_egitim = X[:6000]
X_test = X[6000:]
y_egitim = y[:6000]
y_test = y[6000:]

model = GaussianNB()

model.fit(X_egitim, y_egitim)

y_pred = model.predict(X_test)

print(y_pred)
print(y_test)

dogru_sayi = 0
for index in range(len(y_test)):
    if y_pred[index] == y_test[index]:
        dogru_sayi += 1

print("Test setinde doğru tahmin oranı:", "%",
      (dogru_sayi / len(y_test)) * 100)
