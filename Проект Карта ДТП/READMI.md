[кейс7спринтаКартаДТП.py](https://github.com/user-attachments/files/27896227/7.py)

# **Что нужно сделать**
# 
# Необходимо проверить, встречаются ли в данных дубликаты и пропуски. Это поможет заказчикам собирать более качественные данные.
# Также понадобится ответить на следующие вопросы:
# - как менялось число ДТП по временным промежуткам;
# - различается ли число ДТП для групп водителей с разным стажем.

# **Описание данных**
# 
# Описание данных
# **Датасеты Kirovskaya_oblast.csv, Moscowskaya_oblast.csv содержат информацию о ДТП:
# geometry.coordinates — координаты ДТП;**
# - id — идентификатор ДТП;
# - properties.tags — тег происшествия;
# - properties.light — освещённость;
# - properties.point.lat — широта;
# - properties.point.long — долгота;
# - properties.nearby — ближайшие объекты;
# - properties.region — регион;
# - properties.scheme — схема ДТП;
# - properties.address — ближайший адрес;
# - properties.weather — погода;
# - properties.category — категория ДТП;
# - properties.datetime — дата и время ДТП;
# - properties.injured_count — число пострадавших;
# - properties.parent_region — область;
# - properties.road_conditions — состояние покрытия;
# - properties.participants_count — число участников;
# - properties.participant_categories — категории участников.
# 
# 
# **Датасеты Moscowskaya_oblast_participiants.csv, Kirovskaya_oblast_participiants.csv хранят сведения об участниках ДТП:**
# - role — роль;
# - gender — пол;
# - violations — какие правила дорожного движения были нарушены конкретным участником;
# - health_status — состояние здоровья после ДТП;
# - years_of_driving_experience — число лет опыта;
# - id — идентификатор ДТП.
# 
# 
# **Датасеты Kirovskaya_oblast_vehicles.csv, Moscowskaya_oblast_vehicles.csv хранят сведения о транспортных средствах:**
# - year — год выпуска;
# - brand — марка транспортного средства;
# - color — цвет;
# - model — модель;
# - category — категория;
# - id — идентификатор ДТП.
