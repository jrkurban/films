# Movie Intel

IMDb ve benzeri kaynaklardan gelen büyük film tablolarını **tek ortak şemada** birleştirir, bellek dostu biçimde küçültür ve geçmiş yayın reytingleriyle **yeni bir filmin tahmini puanını** üretir. Aynı model, aday vizyon tarihi/saatlerini tarayarak **en yüksek reytingi getirecek zamanlamayı** önerir.

Bu depo gerçek IMDb dökümlerini indirmek zorunda değildir. `generate-sample` komutu IMDb TSV + bütçe CSV + yayınlanmış reyting CSV üretir; kendi dosyalarınızı aynı komutlarla bağlayabilirsiniz.

## Neden zaman serisi değil, gradyan artırma?

Her film bir kez vizyona girer. Elinizdeki gözlem, aynı filmin saat saat evrilen bir serisi değil; **farklı filmlerin tek seferlik sonuçlarıdır**. Bu yüzden ARIMA / Prophet gibi klasik zaman serisi modelleri hedefe uymaz.

Doğru çerçeve **tabular regresyon**dur:

| Model | Rol |
| --- | --- |
| **HistGradientBoostingRegressor** | Ağaç tabanlı aday. Karışık özellikler, eksik değerler ve bütçe × tür × takvim etkileşimleri için. |
| **Ridge** | Lineer taban çizgisi. Eğitimde test RMSE’si daha düşük olan model üretime alınır. |
| **Takvim özellikleri** | Ay, haftanın günü, saat, hafta sonu, yaz / ödül sezonu, ocak “dump” penceresi. Zaman bilgisi buradan girer. |
| **Hedef kodlama** | Yönetmen, oyuncu, dil, ülke için eğitim setinde yumuşatılmış ortalama reyting. Test sızıntısı yoktur. |

Eğitim **zamansal** ayrılır: eski yıllar train, yeni yıllar test. Böylece model “geleceği” geçmişten tahmin eder.

## Bellek stratejisi

- CSV/TSV/gzip **parça parça** okunur (`chunksize`, varsayılan 50_000).
- Kullanılmayan kolonlar erken düşer; sayılar `float32` / `Int32` olur.
- Ara çıktılar **ZSTD Parquet** satır gruplarıdır; tüm ham tablo RAM’e alınmaz.
- Çakışan kayıtlar önce `imdb_id`, sonra `(normalize başlık + yıl)` ile birleşir.
- Reytingler oy sayısına göre **ağırlıklı ortalama**lanır.

Gerçek IMDb dosyaları (`title.basics.tsv.gz` vb.) aynı boru hattına verilebilir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Çalıştırma

```bash
# 1) Örnek kaynaklar
movie-intel generate-sample --n-movies 6000

# 2) Birleştir, temizle, küçült
movie-intel merge --from-sample

# 3) Modeli eğit
movie-intel train

# 4) Tek tahmin
movie-intel predict \
  --title "Mavi Saat İzmir" \
  --release 2026-11-20T19:00 \
  --director "Nuri Bilge Ceylan" \
  --cast "Haluk Bilginer, Tilda Swinton" \
  --genres Drama \
  --budget 8500000 \
  --runtime 126

# 5) Vizyon penceresi öner
movie-intel recommend \
  --title "Mavi Saat İzmir" \
  --window-start 2026-11-01 \
  --window-end 2026-12-15 \
  --director "Nuri Bilge Ceylan" \
  --genres Drama \
  --budget 8500000

# 6) Arayüz
movie-intel serve --port 43147
```

Kendi dosyalarınız:

```bash
movie-intel merge data/raw/imdb/title.basics.tsv data/raw/imdb/title.ratings.tsv data/raw/published/ratings.csv
```

Kolon adları esnektir (`primaryTitle`, `averageRating`, `production_budget`, `premiere` …). Eşleme `movie_intel/schema.py` içindedir.

## Örnek tahmin fonksiyonu

```python
from movie_intel.model import MovieRatingModel, predict_rating, recommend_release_slots

model = MovieRatingModel.load("models/rating_model.joblib")

movie = {
    "title": "Mavi Saat İzmir",
    "director": "Nuri Bilge Ceylan",
    "cast": "Haluk Bilginer, Tilda Swinton",
    "genres": "Drama",
    "budget_usd": 8_500_000,
    "runtime_min": 126,
    "language": "tr",
    "country": "TR",
    "release_date": "2026-11-20",
    "release_hour": 19,
}

score = predict_rating(model, movie)
slots = recommend_release_slots(model, movie, "2026-11-01", "2026-12-15", top_n=8)
```

`recommend_release_slots` filmin diğer özelliklerini sabit tutup aday gün/saat ızgarasında reytingi yeniden tahmin eder; en yüksek skorlu dilimleri döner.

## Proje düzeni

```
movie_intel/     birleştirme, temizlik, özellikler, model, CLI
webapp/          tahmin ve zamanlama arayüzü
tests/           birleştirme + eğitim duman testi
data/            ham / ara / birleşik parquet (üretilir)
models/          joblib model ve metrikler
```

## Test

```bash
pytest
```
