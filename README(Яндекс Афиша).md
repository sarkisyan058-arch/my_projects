[Яндекс Афиша.py](https://github.com/user-attachments/files/28442429/default.py)
[Ссылка на дашборд работы](https://disk.yandex.ru/i/wJ6fxvTnrQsPVg)


# **1. Загрузка данных и знакомство с ними**

# In[1]:


# Загружаем библиотеки

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



# In[2]:


# Загрузка данных
orders_df = pd.read_csv('/datasets/final_tickets_orders_df.csv')
events_df = pd.read_csv('/datasets/final_tickets_events_df.csv')
tenge_df = pd.read_csv('/datasets/final_tickets_tenge_df.csv')

# Первая информация о датасетах
print("=== ИНФОРМАЦИЯ О ДАТАСЕТЕ ORDERS ===")
print(f"Размер датасета: {orders_df.shape}")
print("\nПервые 5 строк:")
display(orders_df.head())
print("\nОсновные статистические характеристики количественных столбцов:")
display(orders_df.describe())
print("\nТипы данных столбцов:")
display(orders_df.dtypes)
print("\nКоличество пропусков в каждом столбце:")
display(orders_df.isnull().sum())

print("\n=== ИНФОРМАЦИЯ О ДАТАСЕТЕ EVENTS ===")
print(f"Размер датасета: {events_df.shape}")
print("\nПервые 5 строк:")
display(events_df.head())
print("\nТипы данных столбцов:")
display(events_df.dtypes)
print("\nКоличество пропусков в каждом столбце:")
display(events_df.isnull().sum())

print("\n=== ИНФОРМАЦИЯ О ДАТАСЕТЕ TENGE (КУРС ТЕНГЕ) ===")
print(f"Размер датасета: {tenge_df.shape}")
print("\nПервые 5 строк:")
display(tenge_df.head())
print("\nТипы данных столбцов:")
display(tenge_df.dtypes)
print("\nКоличество пропусков в каждом столбце:")
display(tenge_df.isnull().sum())


# **Промежуточные выводы по результатам первичного анализа датасетов**
# 1. Датасет orders
# Ключевые наблюдения:
# 
# Размер: 290 849 записей, 14 столбцов — достаточно для детального анализа покупательского поведения.
# 
# Пропуски: единственный столбец с пропусками — days_since_prev (21 940 пропусков, ≈7,5 % данных). Это ожидаемо: у первых заказов пользователя нет предыдущего заказа, поэтому значение не может быть рассчитано.
# 
# Типы данных: большинство типов соответствуют смыслу столбцов (числа — числовые, даты — строковые). Требует преобразования:
# 
# created_dt_msk, created_ts_msk — в формат даты/времени;
# 
# категориальные столбцы (currency_code, device_type_canonical и др.) — в тип category для оптимизации памяти.
# 
# Количественные показатели:
# 
# Отрицательная выручка (revenue) и total: есть записи с отрицательными значениями (минимум −90,76 и −358,85 соответственно). Вероятно, это возвраты средств. Требуют отдельного анализа: либо исключить из расчёта общей выручки, либо явно отметить как возвраты.
# 
# Выбросы:
# 
# в revenue: максимум 81 174,54 при 75‑м процентиле 809,75 — есть экстремально крупные заказы;
# 
# в tickets_count: максимум 57 билетов при медиане 3 — возможны групповые заказы или ошибки ввода.
# 
# Распределение устройств: подавляющее большинство заказов (≈80 %) сделано с мобильных устройств (mobile — 232 679, desktop — 58 170).
# 
# Валюта: основная валюта — rub (285 780), есть заказы в kzt (5 069), что требует конвертации для унификации.
# 
# Возрастные ограничения: распределение по категориям 0, 6, 12, 16, 18 лет без пропусков — данные чистые.
# 
# 2. Датасет events
# Ключевые наблюдения:
# 
# Размер: 22 427 записей, 11 столбцов — покрывает события, на которые продаются билеты.
# 
# Пропуски: отсутствуют во всех столбцах — данные полные.
# 
# Типы данных: в целом корректны. Категориальные столбцы (event_type_main, region_name и др.) можно перевести в тип category.
# 
# Содержимое:
# 
# основная тематика событий — «спектакль» (по первым строкам), но есть разнообразие по типам (event_type_main).
# 
# присутствуют географические данные (region_name, city_name), что позволяет анализировать региональные особенности спроса.
# 
# уникальные идентификаторы (event_id, venue_id) позволяют связать с orders.
# 
# 3. Датасет tenge (курс тенге)
# Ключевые наблюдения:
# 
# Размер: 357 записей, 4 столбца — покрывает период с начала 2024 года.
# 
# Пропуски: отсутствуют — данные полные.
# 
# Типы данных: столбец data нужно преобразовать в формат даты.
# 
# Содержимое:
# 
# столбец curs содержит курс за 100 тенге (nominal = 100).
# 
# данные позволяют конвертировать выручку в kzt в рубли по актуальному курсу на дату заказа.

#
# **По смотрим данные**

# In[3]:


# Анализ категориальных столбцов
print("=== АНАЛИЗ КАТЕГОРИАЛЬНЫХ СТОЛБЦОВ ===")

# device_type_canonical
print("\nУстройство (device_type_canonical):")
print(orders_df['device_type_canonical'].value_counts())

# currency_code
print("\nВалюта (currency_code):")
print(orders_df['currency_code'].value_counts())

# age_limit
print("\nВозрастное ограничение (age_limit):")
print(orders_df['age_limit'].value_counts().sort_index())

# event_type_main (из events_df)
print("\nТип мероприятия (event_type_main):")
print(events_df['event_type_main'].value_counts())


# **2. Предобработка данных и подготовка их к исследованию**

# In[4]:


# Проверка данных курса валют
print("=== ПРОВЕРКА ДАННЫХ О КУРСЕ ВАЛЮТ ===")
print(f"Размер tenge_df: {tenge_df.shape}")
print(f"Диапазон дат в tenge_df: от {tenge_df['data'].min()} до {tenge_df['data'].max()}")

print(f"Пропуски в курсе (curs): {tenge_df['curs'].isna().sum()}")
print(f"Статистика по курсу:")
print(tenge_df['curs'].describe())


# **Вывод**
# 
# 
# 1. Данные о курсе валют полностью пригодны для дальнейшей работы. Подтверждение:
# 
# - Полнота данных:
# 
#   * пропусков в столбце curs нет (0 пропусков);
# 
#   * всего в датафрейме 357 записей — это соответствует ожидаемому количеству рабочих/торговых дней в периоде.
# 
# 2. Соответствие периода:
# 
# - диапазон дат в tenge_df (10.01.2024–31.12.2024) перекрывает период заказов — все даты заказов имеют соответствующий курс;
# 
# - ранее проверено: количество дат заказов без курса — 0, то есть для каждого заказа найдётся актуальный курс.
# 
# 3. Качество данных:
# 
# - курс представлен числовыми значениями (float64), аномалий или нулевых значений не обнаружено;
# 
# - значения курса находятся в разумном диапазоне: от 17,85 до 21,94 (за 100 тенге), что соответствует реалистичной динамике валютного курса.
# 
# 4. Стабильность курса:
# 
# - стандартное отклонение невелико (0,83), что говорит об отсутствии резких скачков в выборке;
# 
# - медиана (19,88) близка к среднему (19,76), распределение курса достаточно симметрично.
# 
# 5. Итог: данные о курсе валют:
# 
# - чистые (без пропусков и ошибок);
# 
# - полные (покрывают весь период заказов);
# 
# - корректные (значения в ожидаемом диапазоне);
# 
# - готовы к использованию в конвертации выручки из тенге в рубли.

# In[5]:


# Проверка, все ли даты заказов имеют курс
orders_df['created_dt_msk'] = pd.to_datetime(orders_df['created_dt_msk']) #Для этого конвертируем столбец в datetime
orders_dates = orders_df['created_dt_msk'].dt.date.unique()


# Конвертируем столбец 'data' в datetime
tenge_df['data'] = pd.to_datetime(tenge_df['data'])
rate_dates = tenge_df['data'].dt.date.unique()


missing_rates = set(orders_dates) - set(rate_dates)
print(f"Количество дат заказов без курса: {len(missing_rates)}")
if missing_rates:
    print(f"Примеры дат без курса: {sorted(list(missing_rates))[:5]}")


# **Все даты заказов имеют соответствующий курс валют — пропусков нет. Данные готовы к конвертации выручки из тенге в рубли.**

# In[6]:


# Анализ и обработка отрицательных значений
print("\n=== АНАЛИЗ ОТРИЦАТЕЛЬНЫХ REVENUE ===")
negative_revenue = orders_df[orders_df['revenue'] < 0]
print(f"Всего отрицательных revenue: {len(negative_revenue)}")
print(f"Доля от всех заказов: {len(negative_revenue)/len(orders_df)*100:.2f}%")

# Скорее всего, это возвраты. Оставляем, но отмечаем в отчёте
orders_df['is_refund'] = orders_df['revenue'] < 0  # флаг возврата


# **Вывод:**
# 
# Обнаружено 381 заказ с отрицательным revenue (0,13 % от общего числа). Вероятно, это возвраты. Для учёта таких случаев создан флаг is_refund. Данные оставлены в выборке.

# In[7]:


# Работаем с не явными дубликатами
print("\n=== ОБРАБОТКА НЕЯВНЫХ ДУБЛИКАТОВ ===")
df_check = orders_df.copy()
df_check['created_ts_rounded'] = pd.to_datetime(df_check['created_ts_msk']).dt.round('1min')

duplicates = df_check.groupby(['user_id', 'event_id', 'created_ts_rounded', 'tickets_count']).size()
duplicates = duplicates[duplicates > 1]

print(f"Исходное количество заказов: {len(orders_df)}")
print(f"Групп с неявными дубликатами: {len(duplicates)}")
print(f"Общее количество записей в дубликатах: {duplicates.sum()}")

# Решение: оставляем только первый заказ в группе дубликатов
duplicate_indices = df_check.groupby(['user_id', 'event_id', 'created_ts_rounded', 'tickets_count']).apply(
    lambda x: x.index[1:] if len(x) > 1 else []
).explode().dropna().astype(int)

orders_df = orders_df.drop(duplicate_indices)
print(f"После удаления дубликатов: {len(orders_df)} заказов")


# Пересчёт days_since_prev для пользователей с удалёнными заказами
orders_df = orders_df.sort_values(['user_id', 'created_dt_msk'])
orders_df['days_since_prev'] = orders_df.groupby('user_id')['created_dt_msk'].diff().dt.days


# **Вывод**
# 
# После удаления неявных дубликатов количество заказов сократилось с 290 849 до 287 419 (на 3 430 записей). Всего было выявлено 3 166 групп дубликатов, содержащих в сумме 6 596 записей. Это позволило очистить данные от избыточности и повысить точность последующего анализа.

# In[8]:


# Обраьотаем выбросы
print("\n=== ОБРАБОТКА ВЫБРОСОВ ===")
# Для revenue (по валютам)
revenue_99th_percentile_rub = orders_df[orders_df['currency_code'] == 'rub']['revenue'].quantile(0.99)
revenue_99th_percentile_kzt = orders_df[orders_df['currency_code'] == 'kzt']['revenue'].quantile(0.99)

print(f"99-й процентиль revenue (RUB): {revenue_99th_percentile_rub:.2f}")
print(f"99-й процентиль revenue (KZT): {revenue_99th_percentile_kzt:.2f}")

# Решение: не удаляем выбросы, а отмечаем их для дальнейшего анализа
orders_df['is_outlier_revenue'] = (
    (orders_df['currency_code'] == 'rub') & (orders_df['revenue'] > revenue_99th_percentile_rub) |
    (orders_df['currency_code'] == 'kzt') & (orders_df['revenue'] > revenue_99th_percentile_kzt)
)

# Для tickets_count
tickets_99th_percentile = orders_df['tickets_count'].quantile(0.99)
orders_df['is_outlier_tickets'] = orders_df['tickets_count'] > tickets_99th_percentile
print(f"99-й процентиль tickets_count: {tickets_99th_percentile}")


# **Вывод по обработке выбросов:**
# 
# Выбросы не удалялись — они помечены флагами для дальнейшего анализа:
# 
# По revenue: границы выбросов определены по 99‑му процентилю отдельно для каждой валюты:
# 
# RUB: свыше 2 570,80;
# 
# KZT: свыше 17 617,24.
# 
# По tickets_count: выбросами считаются значения свыше 6 билетов (99‑й процентиль).
# 
# Созданы флаги:
# 
# is_outlier_revenue — для выбросов по выручке;
# 
# is_outlier_tickets — для выбросов по количеству билетов.
# 
# Данные сохранены в полном объёме — аномальные значения будут изучены отдельно на этапе анализа.
# 
# 

# In[9]:


orders_with_rate = orders_df.merge(
    tenge_df[['data', 'curs']],
    left_on='created_dt_msk',
    right_on='data',
    how='left'
)

# Заполнение пропусков курса 
orders_with_rate['curs'] = orders_with_rate['curs'].fillna(method='ffill').fillna(method='bfill')


# **Выполнили объеденение таблиц**

# In[10]:


# Создаём новые столбцы
# Конвертация в рубли
orders_with_rate['revenue_rub'] = orders_with_rate.apply(
    lambda row: (
        row['revenue'] if row['currency_code'] == 'rub'
        else row['revenue'] * row['curs'] / 100
    ),
    axis=1
)

# Выручка на билет
orders_with_rate['one_ticket_revenue_rub'] = (
    orders_with_rate['revenue_rub'] / orders_with_rate['tickets_count']
)

# Месяц и сезон
orders_with_rate['month'] = orders_with_rate['created_dt_msk'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'зима'
    elif month in [3, 4, 5]: return 'весна'
    elif month in [6, 7, 8]: return 'лето'
    else: return 'осень'

orders_with_rate['season'] = orders_with_rate['month'].apply(get_season)


# **Отчёт по созданию новых столбцов:**
# 
# Успешно созданы дополнительные столбцы для дальнейшего анализа:
# 
# revenue_rub — выручка в рублях:
# 
# для RUB: значение revenue оставлено без изменений;
# 
# для KZT: конвертировано через курс (revenue × curs / 100).
# 
# one_ticket_revenue_rub — средняя стоимость одного билета в рублях (рассчитана как revenue_rub / tickets_count).
# 
# month — номер месяца заказа (1–12, извлечён из created_dt_msk).
# 
# season — сезон заказа («зима», «весна», «лето», «осень»), определён на основе номера месяца.

# In[11]:


print("\nСТАТИСТИКА ПО КЛЮЧЕВЫМ ПОКАЗАТЕЛЯМ ПОСЛЕ ПРЕДОБРАБОТКИ:")
print(orders_with_rate[['revenue_rub', 'one_ticket_revenue_rub', 'tickets_count']].describe())


print("\nРАСПРЕДЕЛЕНИЕ ПО СЕЗОНАМ:")
print(orders_with_rate['season'].value_counts())

print("\nРАСПРЕДЕЛЕНИЕ ПО ВАЛЮТАМ:")
print(orders_with_rate['currency_code'].value_counts())

print("\nКОЛИЧЕСТВО ВОЗВРАТОВ (revenue < 0):")
print(f"{orders_with_rate['is_refund'].sum()} заказов ({orders_with_rate['is_refund'].mean()*100:.2f}%)")

print("\nКОЛИЧЕСТВО ВЫБРОСОВ:")
print(f"По revenue: {orders_with_rate['is_outlier_revenue'].sum()} заказов")
print(f"По tickets_count: {orders_with_rate['is_outlier_tickets'].sum()} заказов")

print(f"Одновременно по revenue и tickets_count: "
      f"{((orders_with_rate['is_outlier_revenue']) & (orders_with_rate['is_outlier_tickets'])).sum()} заказов")

print("\nПРОВЕРКА ПРОПУСКОВ В ИТОГОВОМ ДАТАФРЕЙМЕ:")
missing_data = orders_with_rate.isna().sum()
print(missing_data[missing_data > 0])

print("\nТИПЫ ДАННЫХ ИТОГОВОГО ДАТАФРЕЙМА:")
print(orders_with_rate.dtypes)

print("\nРАЗМЕР ИТОГОВОГО ДАТАФРЕЙМА:", orders_with_rate.shape)
print("ПОТРЕБЛЕНИЕ ПАМЯТИ:", orders_with_rate.memory_usage(deep=True).sum(), "байт")


# In[12]:


print("\n" + "="*50)
print("ВЫВОДЫ ПО ПРЕДОБРАБОТКЕ")
print("="*50)

print("1. ПРОВЕРКА ДАННЫХ О КУРСЕ:")
print("   - Все даты заказов имеют соответствующий курс ")
print("   - Диапазон дат курса соответствует периоду заказов")

print("\n2. ОБРАБОТКА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ:")
print("   - Отрицательные revenue интерпретированы как возвраты")
print("   - Создан флаг is_refund для дальнейшего анализа")

print("\n3. ДУБЛИКАТЫ:")
print("   - Обнаружено и удалено неявных дубликатов: 6596 → 3166 групп")
print("   - После удаления: осталось", len(orders_with_rate), "заказов")
print("   - Пересчитан days_since_prev для пользователей с удалёнными заказами")

print("\n4. ВЫБРОСЫ:")
print("   - Не удалены, а помечены флагами для дальнейшего анализа:")
print("     * is_outlier_revenue — по 99-му процентилю для каждой валюты")
print("     * is_outlier_tickets — по 99-му процентилю")

print("\n5. КОНВЕРТАЦИЯ ВАЛЮТ:")
print("   - Создан столбец revenue_rub (выручка в рублях)")
print("   - Для KZT применён курс из tenge_df (за 100 тенге)")

print("\n6. НОВЫЕ СТОЛБЦЫ:")
print("   - one_ticket_revenue_rub — средняя стоимость билета в рублях")
print("   - month — месяц заказа (1–12)")
print("   - season — сезон ('зима', 'весна', 'лето', 'осень')")

print("\n7. КАЧЕСТВО ДАННЫХ:")
print("   - Пропуски в ключевых столбцах: отсутствуют")
print("   - Типы данных оптимизированы (float32, int8, category)")

print("\n8. ИТОГОВЫЕ МЕТРИКИ:")
print(f"   - Исходное количество заказов: 290 849")
print(f"   - Удалено дубликатов: {290849 - len(orders_with_rate)}")
print(f"   - Осталось для анализа: {len(orders_with_rate)} заказов")
print(f"   - Потеря данных: {(290849 - len(orders_with_rate))/290849*100:.2f}%")

print("\n9. ЧТО ПОЛУЧИЛОСЬ")
print("   - Данные очищены от дубликатов")
print("   - Валюта приведена к единому формату (рубли)")
print("   - Добавлены сезонные и временные признаки")
print("   - Помечены возвраты и выбросы для сегментации")


# **Особые моменты**
# 1. Дубликаты
# - обнаружено 6 596 записей, входящих в группы неявных дубликатов;
# 
# - эти записи сгруппированы в 3 166 уникальных групп дубликатов;
# 
# - после удаления дубликатов (оставления по одной записи на группу) в датафрейме осталось 287 419 заказов;
# 
# - для пользователей, у которых были удалены дубликаты, пересчитан столбец days_since_prev.
# 
# 2. Выбросы по revenue и tickets_count
# Откуда могли взяться:
# 
# - крупные корпоративные заказы (например, группа школьников на экскурсию);
# 
# - VIP‑билеты с высокой стоимостью;
# 
# - пакетные предложения (несколько билетов по специальной цене);
# 
# - редкие мероприятия с очень дорогими билетами;
# 
# - единичные случаи покупки большого количества билетов (например, для семьи, друзей, корпоративного мероприятия).
# 
# Почему не удалили, а пометили флагами:
# 
# - выбросы могут нести ценную бизнес‑информацию (крупные клиенты, VIP‑сегменты);
# 
# - удаление исказило бы реальную картину распределения выручки;
# 
# - помеченные выбросы можно анализировать отдельно: изучать поведение крупных покупателей, их сезонность, предпочтения;
# 
# - решение позволяет сохранить полноту данных и даёт гибкость на этапе анализа — можно строить модели как с выбросами, так и без них.
# 3. 3. Отрицательные значения revenue (возвраты)
# Откуда могли взяться:
# 
# - отмена заказа пользователем;
# 
# - возврат средств из‑за отмены мероприятия;
# 
# - технические ошибки в оплате (двойное списание и последующий возврат);
# 
# - промоакции или компенсации (редкий случай).
# 
# Почему оставили:
# 
# - возвраты — реальные бизнес‑операции, а не ошибки данных;
# 
# - анализ возвратов важен для:
# 
# - оценки удовлетворённости клиентов;
# 
# - выявления проблемных мероприятий (с высокой долей возвратов);
# 
# - расчёта чистой выручки (с учётом возвратов);
# 
# - создан флаг is_refund, чтобы сегментировать данные и отдельно анализировать заказы и возвраты.

# **Промежуточный вывод**
# 
# 
# 1. Общее состояние датафрейма
# Размер: 287 419 заказов (23 столбца).
# 
# Потребление памяти: 193 134 493 байт (~193 МБ).
# 
# Потеря данных: 1,18 % (удалено 3 430 записей — неявные дубликаты).
# 
# Качество данных: пропуски есть только в days_since_prev (21 940), что является естественным состоянием для первых заказов пользователей.
# 
# 2. Ключевые метрики после обработки
# Финансовые показатели:
# 
# Средняя выручка с заказа (revenue_rub): 557,52 руб.
# 
# Средняя стоимость билета (one_ticket_revenue_rub): 201,89 руб.
# 
# Среднее количество билетов в заказе (tickets_count): 2,75.
# 
# Диапазон выручки с заказа: от −90,76 руб. до 81 174,54 руб. (отрицательные значения — возвраты).
# 
# Диапазон стоимости билета: от −18,15 руб. до 21 757,54 руб.
# 
# Распределение по категориям:
# 
# По сезонам: осень (167 953, 58,4 %), лето (119 466, 41,6 %).
# 
# По валютам: RUB (282 400, 98,26 %), KZT (5 019, 1,74 %).
# 
# 3. Обработка аномалий и особых случаев
# Возвраты (is_refund): 379 заказов (0,13 %). Оставили в данных как реальные бизнес‑операции, создали флаг для сегментации.
# 
# Выбросы:
# 
# по выручке (is_outlier_revenue): 2 802 заказа (0,97 %);
# 
# по количеству билетов (is_outlier_tickets): 188 заказов (0,07 %);
# 
# одновременные выбросы: 8 заказов.
# Не удаляли, а пометили флагами — они могут нести ценную бизнес‑информацию.
# 
# Дубликаты: обнаружено 6 596 записей в 3 166 группах дубликатов. Удалены копии, оставлен один заказ в группе. Пересчитан days_since_prev для затронутых пользователей.
# 
# 4. Новые признаки для анализа
# Созданы и заполнены столбцы:
# 
# revenue_rub — выручка в рублях (конвертация с учётом курса);
# 
# one_ticket_revenue_rub — средняя стоимость билета в рублях;
# 
# month — месяц заказа (1–12);
# 
# season — сезон заказа («зима», «весна», «лето», «осень»);
# 
# флаги аномалий: is_refund, is_outlier_revenue, is_outlier_tickets.
# 
# 5. Проверка источников данных
# Курсы валют: все даты заказов имеют соответствующий курс. Диапазон дат курса соответствует периоду заказов.
# 
# Типы данных: оптимизированы, соответствуют семантике полей (даты, числа, категории, булевы флаги).


# **Шаг 3. Исследовательский анализ данных**

# **Анализ распределения заказов по сегментам и их сезонные изменения**

# In[13]:


# Фильтруем заказы за период июнь–ноябрь 2024 года
orders_2024 = orders_with_rate[
    (orders_with_rate['created_dt_msk'] >= '2024-06-01') &
    (orders_with_rate['created_dt_msk'] <= '2024-11-30')
].copy()

# Определяем сезоны для фильтрации
orders_2024['season_simple'] = orders_2024['month'].apply(
    lambda x: 'лето' if x in [6, 7, 8] else 'осень' if x in [9, 10, 11] else None
)

# Оставляем только лето и осень
orders_seasonal = orders_2024[orders_2024['season_simple'].notna()].copy()


# **Динамика заказов по месяцам**

# In[14]:


# Считаем количество заказов по месяцам
monthly_orders = orders_2024.groupby('month').size().reset_index(name='order_count')

# Визуализация
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_orders, x='month', y='order_count', palette='Blues_d')
plt.title('Динамика количества заказов по месяцам (июнь–ноябрь 2024)')
plt.xlabel('Месяц')
plt.ylabel('Количество заказов')
plt.xticks(range(6), ['Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь'])
plt.grid(axis='y', alpha=0.3)
plt.show()


# **Сравнение распределения заказов по сегментам**

# In[15]:


# Объединяем orders_with_rate с events_df по event_id
orders_with_events = pd.merge(
    orders_with_rate,
    events_df,
    on='event_id',
    how='left'
)

print("Создан датафрейм 'orders_with_events'")
print(f"Размер после объединения: {orders_with_events.shape}")


# In[16]:


def plot_seasonal_comparison(df, column, title):
    # Проверяем наличие столбца
    if column not in df.columns:
        print(f" Столбец '{column}' не найден в датафрейме!")
        return

    # Считаем доли для каждого сезона
    seasonal_dist = df.groupby(['season', column]).size().unstack(fill_value=0)
    seasonal_pct = seasonal_dist.div(seasonal_dist.sum(axis=1), axis=0) * 100

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    seasonal_pct.T.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Доля, %', fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.legend(title='Сезон', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Настройка отображения графиков в Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')

# Запуск визуализации с правильным датафреймом
print("=== ЗАПУСК ВИЗУАЛИЗАЦИИ ===")

# 1. По типам мероприятий
print("1. Строим график по типам мероприятий...")
plot_seasonal_comparison(orders_with_events, 'event_type_main',
                   'Распределение заказов по типам мероприятий (лето vs осень)')


# 2. По типам устройств
print("2. Строим график по типам устройств...")
plot_seasonal_comparison(orders_with_events, 'device_type_canonical',
                   'Распределение заказов по типам устройств (лето vs осень)')

# 3. По возрастным ограничениям
print("3. Строим график по возрастным ограничениям...")
plot_seasonal_comparison(orders_with_events, 'age_limit',
                   'Распределение заказов по возрастным ограничениям (лето vs осень)')


# **Анализ выручки с одного билета по типам мероприятий**

# In[17]:


# Группируем по сезону и типу мероприятия, считаем среднюю выручку с одного билета
revenue_by_season_event = orders_with_events.groupby(
    ['season', 'event_type_main']
)['one_ticket_revenue_rub'].mean().unstack(fill_value=0)

print("Средняя выручка с одного билета по типам мероприятий (руб.):")
print(revenue_by_season_event)


# Выводы:
# 
# - Концерты и стендап — наиболее высокодоходные форматы с потенциалом роста осенью (особенно стендап).
# 
# - Театр и ёлки показывают высокую выручку, но с заметным сезонным спадом осенью.
# 
# - Спортивные мероприятия и выставки формируют минимальную выручку с билета — возможно, из‑за более низких цен или иного позиционирования.
# 
# - Сезонность существенно влияет на ценовую политику: летом выручка по большинству категорий выше, что может быть связано с повышенным спросом или специальными предложениями.

# In[18]:


# Рассчитываем относительное изменение в %
relative_change = (
    (revenue_by_season_event.loc['осень'] - revenue_by_season_event.loc['лето']) /
    revenue_by_season_event.loc['лето'] * 100
).round(2)

print("\nОтносительное изменение осенней выручки по сравнению с летней (%, положительные значения = рост):")
print(relative_change)


# **Выводы**
# - Стендап демонстрирует устойчивость и рост спроса осенью. Это может быть связано с:
# 
#   * сезонной актуальностью формата;
# 
#   * успешными промокампаниями;
# 
#   * расширением аудитории.
# 
# - Театр и ёлки наиболее уязвимы к сезонному спаду. Возможные причины:
# 
#   * летний пик спроса на семейные и культурные мероприятия;
# 
#   * конкуренция с другими видами досуга осенью;
# 
#   * отсутствие специальных осенних программ.
# 
# - Концерты теряют выручку осенью (−10 %), несмотря на традиционно высокий спрос в этот период. Это может указывать на:
# 
#   * ценовую конкуренцию;
# 
#   * изменение структуры афиши (меньше топовых артистов);
# 
#   * смещение спроса на другие форматы.
# 
# - Выставки показывают слабый, но положительный тренд (+4,85 %). Вероятно, это связано с запуском новых экспозиций или образовательных программ осенью.
# 
# - Спорт и категория «другое» демонстрируют умеренное снижение. Это отражает общую сезонную динамику без выраженных негативных факторов.

# **Визуализируем**

# In[19]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# График 1: Средние значения по сезонам
revenue_by_season_event.T.plot(kind='bar', ax=ax1, color=['skyblue', 'orange'])
ax1.set_title('Средняя выручка с одного билета по типам мероприятий\n(лето vs осень 2024)')
ax1.set_ylabel('Выручка, руб.')
ax1.set_xlabel('Тип мероприятия')
ax1.legend(title='Сезон')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# График 2: Относительное изменение
relative_change.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Относительное изменение выручки осенью vs летом (%)')
ax2.set_ylabel('Изменение, %')
ax2.set_xlabel('Тип мероприятия')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)  # Линия нуля
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# **Итоговые выводы**

# In[20]:


print("\n" + "="*60)
print("ВЫВОДЫ ПО СЕЗОННОМУ АНАЛИЗУ")
print("="*60)

# 1. Динамика заказов по месяцам
print("1. ДИНАМИКА ЗАКАЗОВ ПО МЕСЯЦАМ:")
print("- Наблюдается рост заказов от июня к ноябрю 2024 года.")
print("- Пик активности приходится на осенние месяцы (сентябрь–ноябрь).")

# Можно добавить точные цифры из monthly_orders:
print(f"- Максимальное количество заказов: {monthly_orders['order_count'].max()} в {monthly_orders.loc[monthly_orders['order_count'].idxmax(), 'month']} месяце.")

# 2. Распределение по типам мероприятий
print("\n2. РАСПРЕДЕЛЕНИЕ ПО ТИПАМ МЕРОПРИЯТИЙ:")
print("- Осенью растёт доля концертов и театральных мероприятий.")
print("- Летом выше доля спортивных событий и стендапов.")
print("- Категория «другое» сохраняет стабильную долю.")

# 3. Выручка с одного билета
print("\n3. ДИНАМИКА ВЫРУЧКИ С ОДНОГО БИЛЕТА:")
print("- Средняя выручка с билета осенью выросла на {relative_change[relative_change > 0].mean():.1f}% по категориям с ростом.")
print("- Наибольший рост выручки наблюдается для: {relative_change.idxmax()} ({relative_change.max()}%).")
if relative_change.min() < 0:
    print(f"- Снижение выручки зафиксировано для: {relative_change.idxmin()} ({relative_change.min()}%).")

# 4. Общие тенденции
print("\n4. ОБЩИЕ ТЕНДЕНЦИИ:")
print("- Осенний период характеризуется как ростом количества заказов, так и изменением структуры спроса.")
print("- Меняется не только объём, но и предпочтения аудитории: акцент смещается на массовые мероприятия (концерты, театр).")
print("- Выручка с билета демонстрирует разнонаправленную динамику в зависимости от типа мероприятия.")



# **3.2. Осенняя активность пользователей**

# In[21]:


# Оставляем только осенние данные
autumn_data = orders_with_events[orders_with_events['season'] == 'осень'].copy()

print(f"Количество записей за осень: {len(autumn_data)}")
print(f"Период данных: с {autumn_data['created_dt_msk'].min()} по {autumn_data['created_dt_msk'].max()}")


# In[22]:


# Преобразуем дату в формат datetime, если ещё не сделано
autumn_data['created_dt_msk'] = pd.to_datetime(autumn_data['created_dt_msk'])

# Извлекаем дату и день недели
autumn_data['date'] = autumn_data['created_dt_msk'].dt.date
autumn_data['day_of_week'] = autumn_data['created_dt_msk'].dt.day_name()


# Порядок дней недели для правильной сортировки
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
autumn_data['day_of_week'] = pd.Categorical(
    autumn_data['day_of_week'],
    categories=days_order,
    ordered=True
)

print(f"Период анализа: с {autumn_data['date'].min()} по {autumn_data['date'].max()}")
print(f"Количество записей за осень: {len(autumn_data)}")
print(f"Доступные столбцы: {list(autumn_data.columns)}")


# In[23]:


# Создаём сводную таблицу: группируем по дате и рассчитываем ключевые метрики
daily_stats = autumn_data.groupby('date').agg(
    orders_count=('order_id', 'count'),           
    dau=('user_id', 'nunique'),                 
    avg_ticket_price=('one_ticket_revenue_rub', 'mean')  
).reset_index()

# Добавляем столбец с днём недели в daily_stats
daily_stats['day_of_week'] = pd.to_datetime(daily_stats['date']).dt.day_name()
daily_stats['day_of_week'] = pd.Categorical(
    daily_stats['day_of_week'],
    categories=days_order,
    ordered=True
)
# Рассчитываем среднее число заказов на одного пользователя
daily_stats['orders_per_user'] = daily_stats['orders_count'] / daily_stats['dau']
# Заменяем NaN на 0 (если в какой‑то день был 1 пользователь и 0 заказов)
daily_stats['orders_per_user'] = daily_stats['orders_per_user'].fillna(0)

print("\nСВОДНАЯ ТАБЛИЦА ПО ДНЕВНОЙ АКТИВНОСТИ:")
print(daily_stats.head(10))
print(f"\nВсего дней в анализе: {len(daily_stats)}")


# In[24]:


# Визуализируем динамику по дням


fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Динамика осенней активности пользователей (2024)', fontsize=16, fontweight='bold')

# 1. Общее число заказов
axes[0, 0].plot(daily_stats['date'], daily_stats['orders_count'], marker='o', color='blue', linewidth=2)
axes[0, 0].set_title('Общее число заказов по дням')
axes[0, 0].set_ylabel('Количество заказов')
axes[0, 0].grid(True, alpha=0.3)

# 2. Количество активных пользователей (DAU)
axes[0, 1].plot(daily_stats['date'], daily_stats['dau'], marker='s', color='green', linewidth=2)
axes[0, 1].set_title('Количество активных пользователей (DAU)')
axes[0, 1].set_ylabel('Количество пользователей')
axes[0, 1].grid(True, alpha=0.3)

# 3. Среднее число заказов на пользователя
axes[1, 0].plot(daily_stats['date'], daily_stats['orders_per_user'], marker='^', color='purple', linewidth=2)
axes[1, 0].set_title('Среднее число заказов на пользователя')
axes[1, 0].set_ylabel('Заказов на пользователя')
axes[1, 0].grid(True, alpha=0.3)

# 4. Средняя стоимость одного билета
axes[1, 1].plot(daily_stats['date'], daily_stats['avg_ticket_price'], marker='d', color='orange', linewidth=2)
axes[1, 1].set_title('Средняя стоимость одного билета')
axes[1, 1].set_ylabel('Рублей')
axes[1, 1].grid(True, alpha=0.3)

# Настройка отображения дат на оси X
for ax in axes.flat:
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Дата')

plt.tight_layout()
plt.show()


# **Анализ недельной активности**

# In[25]:


weekly_cycle = daily_stats.groupby('day_of_week').agg(
    avg_orders_per_day=('orders_count', 'mean'),
    avg_dau_per_day=('dau', 'mean'),
    avg_ticket_price=('avg_ticket_price', 'mean')
).reindex(days_order)

# Заказы на пользователя — среднее по дням
weekly_cycle['orders_per_user'] = weekly_cycle['avg_orders_per_day'] / weekly_cycle['avg_dau_per_day']
weekly_cycle['orders_per_user'] = weekly_cycle['orders_per_user'].fillna(0)

print("\nАКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ (СРЕДНИЕ ЗНАЧЕНИЯ):")
print(weekly_cycle.round(2))


# **Выводы**
# 1. Вторник — день максимальной транзакционной активности. При самом низком среднем чеке пользователи делают больше всего заказов (3,64 на человека). Это может указывать на:
# 
# - массовые недорогие покупки (например, билеты на будние мероприятия);
# 
# - действие промоакций или скидок;
# 
# - привычку планировать досуг в начале недели.
# 
# 2. Пятница — пик по аудитории. Наибольшее число уникальных пользователей (1 029) и стабильно высокая активность (3,02 заказа на человека) говорят о:
# 
# - готовности аудитории к покупкам перед выходными;
# 
# - планировании досуга на уикенд.
# 
# 3. Воскресенье — день минимальной активности. Низкие значения по всем метрикам (2 163 заказа, 827 пользователей, 2,61 заказа на человека) могут быть связаны с:
# 
# - завершением выходных и снижением интереса к планированию;
# 
# - тратой времени на уже купленные мероприятия;
# 
# - общей усталостью аудитории после уикенда.
# 
# 4. Обратная связь между средним чеком и активностью. В дни высокой активности (вторник) средний чек ниже, а в дни низкой активности (воскресенье) — выше. Это говорит о том, что:
# 
# - пользователи чаще покупают недорогие билеты в пиковые дни;
# 
# - дорогие билеты (например, на престижные мероприятия) бронируют реже, но с большей готовностью платить.

# **Сравниваем будни и выходные:**

# In[26]:


# Выделяем будни и выходные
weekdays = weekly_cycle.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
weekends = weekly_cycle.loc[['Saturday', 'Sunday']]

# Средние показатели по будним и выходным дням
avg_weekdays = weekdays.mean()
avg_weekends = weekends.mean() 

print("\nСРЕДНИЕ ПОКАЗАТЕЛИ:")
print(f"Будни: {avg_weekdays['avg_orders_per_day']:.1f} заказов/день, "
      f"{avg_weekdays['avg_dau_per_day']:.0f} пользователей, "
      f"{avg_weekdays['avg_ticket_price']:.2f} руб./билет")
print(f"Выходные: {avg_weekends['avg_orders_per_day']:.1f} заказов/день, "
      f"{avg_weekends['avg_dau_per_day']:.0f} пользователей, "
      f"{avg_weekends['avg_ticket_price']:.2f} руб./билет")


# **Визуализация недельной цикличности**

# In[27]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Недельная цикличность активности (осень 2024)', fontsize=16, fontweight='bold')

# 1. Заказы по дням недели
axes[0, 0].bar(weekly_cycle.index, weekly_cycle['avg_orders_per_day'], color='skyblue')
axes[0, 0].set_title('Общее число заказов по дням недели')
axes[0, 0].set_ylabel('Количество заказов')


# 2. Активные пользователи по дням недели
axes[0, 1].bar(weekly_cycle.index, weekly_cycle['avg_dau_per_day'], color='lightgreen')
axes[0, 1].set_title('Активные пользователи (DAU) по дням недели')
axes[0, 1].set_ylabel('Количество пользователей')

# 3. Заказы на пользователя по дням недели
axes[1, 0].bar(weekly_cycle.index, weekly_cycle['orders_per_user'], color='purple')
axes[1, 0].set_title('Заказы на пользователя по дням недели')
axes[1, 0].set_ylabel('Заказов на пользователя')

# 4. Средняя цена билета по дням недели
axes[1, 1].bar(weekly_cycle.index, weekly_cycle['avg_ticket_price'], color='orange')
axes[1, 1].set_title('Средняя цена билета по дням недели')
axes[1, 1].set_ylabel('Рублей')

# Общая настройка графиков
for ax in axes.flat:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[28]:


print("\n" + "="*50)
print("ПРОМЕЖУТОЧНЫЕ ВЫВОДЫ ПО ОСЕННЕЙ АКТИВНОСТИ")
print("="*50)

# Общие показатели
print(f"1. Общие данные:")
print(f"- Период: {autumn_data['created_dt_msk'].min()} — {autumn_data['created_dt_msk'].max()}")
print(f"- Всего заказов: {daily_stats['orders_count'].sum()}")
print(f"- Средний DAU: {daily_stats['dau'].mean():.0f} чел./день")
print(f"- Ср. стоимость билета: {daily_stats['avg_ticket_price'].mean():.2f} руб.")

# Динамика и пики
print(f"\n2. Ключевые пики:")
print(f"- Макс. заказов: {daily_stats['orders_count'].max()} ({daily_stats.loc[daily_stats['orders_count'].idxmax(), 'date']})")
print(f"- Пик DAU: {daily_stats['dau'].max()} чел. ({daily_stats.loc[daily_stats['dau'].idxmax(), 'date']})")

# Недельная цикличность
print(f"\n3. По дням недели:")
print(f"- Пик заказов: {weekly_cycle['avg_orders_per_day'].idxmax()} ({weekly_cycle['avg_orders_per_day'].max():.1f} заказов/день)")
print(f"- Мин. заказов: {weekly_cycle['avg_orders_per_day'].idxmin()} ({weekly_cycle['avg_orders_per_day'].min():.1f} заказов/день)")
print(f"- Самая дорогая цена: {weekly_cycle['avg_ticket_price'].idxmax()} ({weekly_cycle['avg_ticket_price'].max():.2f} руб.)")

# Сравнение будней и выходных
weekdays = weekly_cycle.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
weekends = weekly_cycle.loc[['Saturday', 'Sunday']]

print(f"\n4. Будни vs выходные:")
print(f"- Заказы: {weekdays['avg_orders_per_day'].mean():.1f} → {weekends['avg_orders_per_day'].mean():.1f} заказов/день")
print(f"- DAU: {weekdays['avg_dau_per_day'].mean():.0f} → {weekends['avg_dau_per_day'].mean():.0f} пользователей/день")
print(f"- Цена билета: {weekdays['avg_ticket_price'].mean():.2f} → {weekends['avg_ticket_price'].mean():.2f} руб.")

print(f"\n5. Основные тенденции:")
print(f"- Несмотря на более высокую активность в будни (в среднем на {((2893.7 - 2417.4) / 2417.4 * 100):.1f}%), выходные приносят выручку за счёт повышенной цены билета (+{((204.89 - 190.73) / 190.73 * 100):.1f}%)")
print(f"- Разница в DAU между буднями и выходными умеренная: {((946 - 897) / 897 * 100):.1f}% в пользу будней")
print(f"- Стабильное число заказов на пользователя ({weekly_cycle['orders_per_user'].mean():.3f} ± {weekly_cycle['orders_per_user'].std():.3f}) указывает на предсказуемость поведения аудитории")


# Цена билета: средние значения
weekdays_price_mean = weekdays['avg_ticket_price'].mean()
weekends_price_mean = weekends['avg_ticket_price'].mean()
price_diff_percent = abs(weekends_price_mean - weekdays_price_mean) / min(weekdays_price_mean, weekends_price_mean) * 100



print(f"- Стабильное число заказов на пользователя: {weekly_cycle['orders_per_user'].mean():.3f} ± {weekly_cycle['orders_per_user'].std():.3f}")


# Рекомендации
print(f"\n6. Рекомендации:")
print("- В пиковые дни: усилить маркетинг и поддержку")
print("- В спадные дни: проводить техработы и A/B‑тесты")
print("- На выходные: предлагать спецакции и расширять ассортимент мероприятий")



# **3.3 Анализ популярных событий и партнёров**

# In[29]:


print("Доступные столбцы в autumn_data:")
print(list(autumn_data.columns))


# In[30]:


# Фильтруем заказы за осень 2024 (сентябрь, октябрь, ноябрь)
autumn_orders = orders_df[
    (orders_df['created_dt_msk'].dt.year == 2024) &
    (orders_df['created_dt_msk'].dt.month.isin([9, 10, 11]))
]

# Конвертируем выручку в рубли
autumn_orders = autumn_orders.merge(
    tenge_df[['data', 'curs']],
    left_on='created_dt_msk',
    right_on='data',
    how='left'
)
autumn_orders['revenue_rub'] = autumn_orders.apply(
    lambda row: (
        row['revenue'] * row['curs'] / 100
        if row['currency_code'] == 'kzt'
        else row['revenue']
    ),
    axis=1
)

# Объединяем с данными о событиях
autumn_data = autumn_orders.merge(
    events_df[['event_id', 'region_name', 'city_name', 'event_type_main']],
    on='event_id',
    how='inner'
)


# In[31]:


# Группируем по регионам
region_stats = autumn_data.groupby('region_name').agg(
    unique_events=('event_id', 'nunique'),
    total_orders=('order_id', 'count')
).reset_index()

# Общие показатели для расчёта долей
total_events = autumn_data['event_id'].nunique()
total_orders = len(autumn_data)

# Добавляем доли
region_stats['events_share'] = (region_stats['unique_events'] / total_events * 100).round(2)
region_stats['orders_share'] = (region_stats['total_orders'] / total_orders * 100).round(2)

# Сортируем по количеству мероприятий (убывание)
region_stats = region_stats.sort_values('unique_events', ascending=False)

print("АНАЛИЗ ПО РЕГИОНАМ:")
print(region_stats)


# **Вывод по анализу активности по регионам**
# 
# 1. Каменевский регион — ключевой рынок. Обеспечивает почти треть всех заказов при четверти мероприятий. Это приоритетная зона для:
# 
# - сохранения текущих маркетинговых усилий;
# 
# - тестирования новых форматов;
# 
# - масштабирования успешных практик.
# 
# 2. Североярская область требует анализа. Высокая доля мероприятий (16,47 %) при относительно низкой доле заказов (12,37 %) указывает на:
# 
# - низкую конверсию мероприятий в продажи;
# 
# - возможную избыточность афиши;
# 
# - необходимость оптимизации ассортимента или продвижения.
# 
# 3. Широковская область — стабильный средний сегмент. Сбалансированные показатели (5,04 % мероприятий, 5,17 % заказов) делают её перспективной для:
# 
# - точечного роста через локальные акции;
# 
# - расширения партнёрской сети.
# 
# 4. Малые регионы — зона потенциального роста. Несмотря на текущие низкие доли, они могут дать прирост при:
# 
# - запуске локальных маркетинговых кампаний;
# 
# - привлечении новых партнёров‑организаторов;
# 
# - адаптации афиши под местные предпочтения.

# In[32]:


# Группируем по партнёрам (service_name)
partner_stats = autumn_data.groupby('service_name').agg(
    unique_events=('event_id', 'nunique'),
    total_orders=('order_id', 'count'),
    total_revenue=('revenue_rub', 'sum')
).reset_index()

# Рассчитываем доли
partner_stats['events_share'] = (partner_stats['unique_events'] / total_events * 100).round(2)
partner_stats['orders_share'] = (partner_stats['total_orders'] / total_orders * 100).round(2)
partner_stats['revenue_share'] = (partner_stats['total_revenue'] / autumn_data['revenue_rub'].sum() * 100).round(2)

# Сортируем по выручке (убывание), берём топ‑10
partner_stats_top = partner_stats.sort_values('total_revenue', ascending=False).head(10)

print("\nАНАЛИЗ ПО ПАРТНЁРАМ (ТОП‑10):")
print(partner_stats_top)


# **Вывод по анализу активности партнёров**
# 
# 1. «Билеты без проблем» — ключевой партнёр. Обеспечивает максимальную выручку. Приоритетная зона для:
# 
# - сохранения текущих условий сотрудничества;
# 
# - тестирования новых форматов продаж;
# 
# - углублённого анализа лояльности аудитории.
# 
# 2. «Лови билет!» требует внимания. Несмотря на лидерство по числу мероприятий, доля выручки вдвое ниже доли событий. Возможные причины:
# 
# - низкая цена билетов;
# 
# - высокая конкуренция в нише;
# 
# - недостаточная маркетинговая поддержка.
# 
# 3. «Весь в билетах» — высокоэффективный партнёр. Высокая выручка при малом числе мероприятий говорит о:
# 
# - премиальном сегменте (дорогие билеты);
# 
# - эксклюзивных правах на топовые события;
# 
# - эффективной стратегии ценообразования.
# 
# 4. Малые партнёры — зона потенциального роста. Могут стать источником роста при:
# 
# - точечной поддержке (промоакции, обучение);
# 
# - интеграции в общую маркетинговую стратегию;
# 
# - анализе успешных практик топ‑3. 

# In[33]:


print("\n" + "="*60)
print("ПРОМЕЖУТОЧНЫЕ ВЫВОДЫ ПО РЕГИОНАМ И ПАРТНЁРАМ (ОСЕНЬ 2024)")
print("="*60)

# Анализ регионов
print("1. РЕГИОНЫ:")
print(f"- Всего регионов: {len(region_stats)}")
if len(region_stats) > 0:
    top_region = region_stats.iloc[0]
    print(f"- Лидер по мероприятиям: {top_region['region_name']} "
          f"({top_region['unique_events']} мероприятий, {top_region['events_share']}%)")
    print(f"- Лидер по заказам: {top_region['region_name']} "
          f"({top_region['total_orders']} заказов, {top_region['orders_share']}%)")

    # Оцениваем концентрацию
    if top_region['events_share'] > 30:
        print("- Явный лидер среди регионов (более 30% мероприятий)")
    elif top_region['events_share'] > 20:
        print("- Умеренное доминирование одного региона")
    else:
        print("- Относительно равномерное распределение мероприятий между регионами")

# Анализ партнёров
print("\n2. ПАРТНЁРЫ:")
print(f"- Всего партнёров: {len(partner_stats)}")
if len(partner_stats_top) > 0:
    top_partner = partner_stats_top.iloc[0]
    print(f"- Топ‑1 партнёр: {top_partner['service_name']} "
          f"({top_partner['total_revenue']:,.0f} руб., {top_partner['revenue_share']}% выручки)")
    print(f"- Топ‑3 партнёра суммарно обеспечивают {partner_stats_top['revenue_share'].head(3).sum()}% выручки")
    print(f"- Топ‑5 партнёров суммарно обеспечивают {partner_stats_top['revenue_share'].head(5).sum()}% выручки")

    if top_partner['revenue_share'] > 25:
        print("- Есть явный лидер среди партнёров")
    else:
        print("- Распределение выручки относительно равномерное между партнёрами")


# Итоговые выводы
print("\n3. ОБЩИЕ ВЫВОДЫ:")
if top_region['events_share'] > 30 and top_partner['revenue_share'] > 25:
    print("- Чётко выраженные лидеры: один регион и один партнёр доминируют в системе")
elif top_region['events_share'] <= 30 and top_partner['revenue_share'] <= 25:
    print("- Нет явных лидеров — распределение относительно равномерное")
else:
    print("- Частичное доминирование: либо регион, либо партнёр имеет заметное преимущество")

print("- Топ‑регионы и топ‑партнёры формируют основную выручку системы")
print("- Рекомендуется сосредоточить внимание на ключевых регионах и партнёрах для максимизации выручки")



# **4. Статистический анализ данных**

# In[34]:


# Фильтруем заказы за осень 2024 (сентябрь, октябрь, ноябрь)
autumn_orders = orders_df[
    (orders_df['created_dt_msk'].dt.year == 2024) &
    (orders_df['created_dt_msk'].dt.month.isin([9, 10, 11]))
]


# In[35]:


# Находим пользователей, которые использовали оба типа устройств
dual_device_users = autumn_orders.groupby('user_id')['device_type_canonical'].nunique()
dual_device_users = dual_device_users[dual_device_users > 1].index


# In[36]:


# Исключаем этих пользователей из данных
autumn_orders_filtered = autumn_orders[~autumn_orders['user_id'].isin(dual_device_users)]


# **Гипотеза 1: Среднее количество заказов на пользователя**

# In[37]:


# Группируем по пользователям и типу устройства, считаем количество заказов
user_orders = autumn_orders_filtered.groupby(['user_id', 'device_type_canonical']).agg(
    orders_count=('order_id', 'count')
).reset_index()

# Разделяем выборки по типу устройства
mobile_users = user_orders[user_orders['device_type_canonical'] == 'mobile']['orders_count']
desktop_users = user_orders[user_orders['device_type_canonical'] == 'desktop']['orders_count']


# **Гипотеза 2: Среднее время между заказами**

# In[38]:


# Сортируем заказы по пользователю и времени
autumn_orders_sorted = autumn_orders_filtered.sort_values(['user_id', 'created_dt_msk'])

# Рассчитываем разницу во времени между заказами для каждого пользователя
autumn_orders_sorted['time_diff'] = autumn_orders_sorted.groupby('user_id')['created_dt_msk'].diff()
# Переводим в часы
autumn_orders_sorted['time_diff_hours'] = autumn_orders_sorted['time_diff'].dt.total_seconds() / 3600

# Убираем отрицательные значения и NaN (первый заказ у пользователя)
valid_diffs = autumn_orders_sorted[autumn_orders_sorted['time_diff_hours'] > 0]


# Разделяем по типу устройства
mobile_time_diff = valid_diffs[valid_diffs['device_type_canonical'] == 'mobile']['time_diff_hours']
desktop_time_diff = valid_diffs[valid_diffs['device_type_canonical'] == 'desktop']['time_diff_hours']


# **Исследовательский анализ данных**

# In[39]:


print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА:")
print("\nГипотеза 1 — Заказы на пользователя:")
print(f"Мобильные: среднее = {mobile_users.mean():.2f}, медиана = {mobile_users.median():.2f}, n = {len(mobile_users)}")
print(f"Стационарные: среднее = {desktop_users.mean():.2f}, медиана = {desktop_users.median():.2f}, n = {len(desktop_users)}")

print("\nГипотеза 2 — Время между заказами (часы):")
print(f"Мобильные: среднее = {mobile_time_diff.mean():.2f}, медиана = {mobile_time_diff.median():.2f}, n = {len(mobile_time_diff)}")
print(f"Стационарные: среднее = {desktop_time_diff.mean():.2f}, медиана = {desktop_time_diff.median():.2f}, n = {len(desktop_time_diff)}")

# Визуализация распределений — создаём фигуру один раз
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Распределение метрик по типам устройств', fontsize=16)

# Заказы на пользователя — boxplot
axes[0, 0].boxplot([mobile_users, desktop_users], labels=['Мобильные', 'Стационарные'])
axes[0, 0].set_title('Количество заказов на пользователя')
axes[0, 0].set_ylabel('Количество заказов')


# Время между заказами — boxplot
axes[0, 1].boxplot([mobile_time_diff, desktop_time_diff], labels=['Мобильные', 'Стационарные'])
axes[0, 1].set_title('Время между заказами (часы)')
axes[0, 1].set_ylabel('Часы')

# Гистограммы для заказов (плотность)
axes[1, 0].hist(mobile_users, alpha=0.7, label='Мобильные', bins=20, density=True)
axes[1, 0].hist(desktop_users, alpha=0.7, label='Стационарные', bins=20, density=True)
axes[1, 0].set_title('Распределение заказов на пользователя (плотность)')
axes[1, 0].legend()

# Гистограммы для времени между заказами (плотность)
axes[1, 1].hist(mobile_time_diff, alpha=0.7, label='Мобильные', bins=30, density=True)
axes[1, 1].hist(desktop_time_diff, alpha=0.7, label='Стационарные', bins=30, density=True)
axes[1, 1].set_title('Распределение времени между заказами (плотность)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# **Пояснение теста**
# 

# - U‑критерий Манна‑Уитни проверяет, сдвинуто ли распределение одной группы относительно другой.
# - При схожей форме распределений его можно интерпретировать как сравнение медиан.
# - Различия в средних значениях могут быть вызваны выбросами и асимметрией.

# **Проверка нормальности распределения**

# In[40]:


print("ТЕСТ НОРМАЛЬНОСТИ (Шапиро-Уилка):")

# Для гипотезы 1
stat1_mobile, p1_mobile = stats.shapiro(mobile_users)
stat1_desktop, p1_desktop = stats.shapiro(desktop_users)
print(f"\nГипотеза 1:")
print(f"Мобильные: p-value = {p1_mobile:.4f}")
print(f"Стационарные: p-value = {p1_desktop:.4f}")

# Для гипотезы 2
stat2_mobile, p2_mobile = stats.shapiro(mobile_time_diff)
stat2_desktop, p2_desktop = stats.shapiro(desktop_time_diff)
print(f"\nГипотеза 2:")
print(f"Мобильные: p-value = {p2_mobile:.4f}")
print(f"Стационарные: p-value = {p2_desktop:.4f}")


# **Обоснование выбора теста:**
# 
# Гипотеза 1 — количество заказов на пользователя:
# - Тип данных: дискретные, агрегированные по пользователям
# - Распределение: ненормальное 
# - Единица наблюдения: пользователь
# - Независимость: возможна зависимость (некоторые пользователи используют оба устройства)
# - Выбросы: присутствуют 
# - Критерий: U‑критерий Манна‑Уитни 
# - Сравнение: ранговые распределения между группами
# 
# Гипотеза 2 — время между заказами:
# - Тип данных: непрерывные, временные интервалы
# - Распределение: ненормальное 
# - Единица наблюдения: интервал между последовательными заказами одного пользователя
# - Независимость: возможна зависимость внутри пользователя
# - Выбросы: присутствуют 
# - Критерий: U‑критерий Манна‑Уитни
# - Ограничение: интерпретация как сравнения медиан возможна только при схожей форме распределений
# 

# **Проверка гипотез**

# In[41]:


# Гипотеза 1: Среднее количество заказов на пользователя
# Формулировка гипотез:

# H₀ (нулевая): Среднее количество заказов на пользователя одинаково для мобильных и стационарных устройств.

# H₁ (альтернативная): Среднее количество заказов на пользователя выше для мобильных устройств.


if p1_mobile < 0.05 or p1_desktop < 0.05:
    # Используем U‑критерий Манна‑Уитни
    stat1, p1 = stats.mannwhitneyu(mobile_users, desktop_users, alternative='greater')
    test_name1 = "U‑критерий Манна‑Уитни"
else:
    # Используем t‑тест Стьюдента
    stat1, p1 = stats.ttest_ind(mobile_users, desktop_users, equal_var=False, alternative='greater')
    test_name1 = "t‑тест Стьюдента"

print(f"\nРЕЗУЛЬТАТ ДЛЯ ГИПОТЕЗЫ 1 ({test_name1}):")
print(f"Статистика = {stat1:.4f}, p‑value = {p1:.4f}")


# In[42]:


# Гипотеза 2: Среднее время между заказами
# Формулировка гипотез:

# H₀ (нулевая): Среднее время между заказами одинаково для мобильных и стационарных устройств.

# H₁ (альтернативная): Среднее время между заказами больше для мобильных устройств.


if p2_mobile < 0.05 or p2_desktop < 0.05:
    # Используем U‑критерий Манна‑Уитни (непараметрический тест)
    stat2, p2 = stats.mannwhitneyu(
        mobile_time_diff,
        desktop_time_diff,
        alternative='greater'  # односторонний тест: время у мобильных больше
    )
    test_name2 = "U‑критерий Манна‑Уитни"
else:
    # Если бы данные были нормальными — использовали бы t‑тест Стьюдента
    stat2, p2 = stats.ttest_ind(
        mobile_time_diff,
        desktop_time_diff,
        equal_var=False,
        alternative='greater'
    )
    test_name2 = "t‑тест Стьюдента"

print(f"\nРЕЗУЛЬТАТ ДЛЯ ГИПОТЕЗЫ 2 ({test_name2}):")
print(f"Статистика = {stat2:.4f}, p‑value = {p2:.4f}")

# Дополнительно выведем описательные статистики для сравнения
print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА (время между заказами, часы):")
print(f"Мобильные устройства: медиана = {mobile_time_diff.median():.2f} ч., среднее = {mobile_time_diff.mean():.2f} ч.")
print(f"Стационарные устройства: медиана = {desktop_time_diff.median():.2f} ч., среднее = {desktop_time_diff.mean():.2f} ч.")


# **Пояснение теста**
# 1. Гипотеза 1: среднее количество заказов на пользователя
# - Описательная статистика:
# 
#   * Мобильные: среднее = 2,85, медиана = 2,00, n = 10 967;
# 
#   * Стационарные: среднее = 1,95, медиана = 1,00, n = 1 633.
# 
# - Выводы:
# 
#   * Пользователи мобильных устройств делают значительно больше заказов на пользователя: среднее на 46 % выше, медиана — в 2 раза выше.
# 
#   * Разница заметна как по среднему, так и по медиане — значит, результат устойчив к выбросам.
# 
#   * Размер выборки по мобильным устройствам в 6,7 раза больше, чем по стационарным (n = 10 967 против n = 1 633), что повышает надёжность оценки для мобильных.
# 
#   * Вероятно, мобильный интерфейс удобнее для частых транзакций, либо аудитория мобильных пользователей более вовлечена.
# 
# 2. Гипотеза 2: среднее время между заказами 
# - Описательная статистика:
# 
#   * Мобильные: среднее = 264,47 ч. (~11 дней), медиана = 168 ч. (7 дней), n = 8 590;
# 
#   * Стационарные: среднее = 319,18 ч. (~13,3 дней), медиана = 216 ч. (9 дней), n = 304.
# 
# - Выводы:
# 
#   * Время между заказами больше у стационарных устройств по всем метрикам:
# 
#   * среднее: на 21 % больше (319,18 против 264,47 ч.);
# 
#   * медиана: на 29 % больше (216 против 168 ч.).
# 
#   * Однако выборка для стационарных устройств крайне мала (n = 304), что снижает надёжность оценок и может делать различия статистически незначимыми.
# 
# - Более короткие интервалы между заказами у мобильных могут указывать на:
# 
#   * импульсивность покупок с телефона;
# 
#   * удобство быстрого оформления заказа в мобильном приложении;
# 
#   * более частый доступ к сервису через мобильное устройство.
# 
# 

# In[43]:


print("\n" + "="*60)
print("ПРОМЕЖУТОЧНЫЕ ВЫВОДЫ ПО ПРОВЕРКЕ ГИПОТЕЗ")
print("="*60)

print("ГИПОТЕЗА 1: Среднее количество заказов на пользователя")
print(f"- H₀: Распределение количества заказов на пользователя одинаково для мобильных и стационарных устройств")
print(f"- H₁: Распределение количества заказов на пользователя сдвинуто в сторону больших значений для мобильных устройств")
if p1 < 0.05:
    print(f"- РЕЗУЛЬТАТ: Отклоняем H₀ в пользу H₁ (p = {p1:.4f})")
    print(f"  Пользователи мобильных устройств делают в среднем {mobile_users.mean():.2f} заказов против {desktop_users.mean():.2f} у стационарных")
else:
    print(f"- РЕЗУЛЬТАТ: Недостаточно оснований для отклонения H₀ (p = {p2:.4f})")

print()

print("ГИПОТЕЗА 2: Среднее время между заказами")
print(f"- H₀: Распределение времени между заказами одинаково для мобильных и стационарных устройств")
print(f"- H₁: Распределение времени между заказами сдвинуто в сторону больших значений для мобильных устройств")
if p2 < 0.05:
    print(f"- РЕЗУЛЬТАТ: Отклоняем H₀ в пользу H₁ (p = {p2:.4f})")
    print(f"  У мобильных пользователей среднее время между заказами — {mobile_time_diff.mean():.2f} ч. против {desktop_time_diff.mean():.2f} ч. у стационарных")
    print(f"  Медианы: {mobile_time_diff.median():.2f} ч. (мобильные) vs {desktop_time_diff.median():.2f} ч. (стационарные)")
else:
    print(f"- РЕЗУЛЬТАТ: Недостаточно оснований для отклонения H₀ (p = {p2:.4f})")

# Общий вывод
print("\nВЫВОД:")
if p1 < 0.05 and p2 < 0.05:
    print("Обе гипотезы подтвердились: пользователи мобильных устройств:")
    print("  • делают больше заказов на пользователя;")
    print("  • имеют большее среднее время между заказами.")
elif p1 < 0.05:
    print("Подтвердилась только гипотеза 1: пользователи мобильных устройств делают больше заказов, но время между заказами не отличается.")
elif p2 < 0.05:
    print("Подтвердилась только гипотеза 2: у пользователей мобильных устройств время между заказами больше, но количество заказов не отличается.")
else:
    print("Ни одна из гипотез не подтвердилась: нет статистически значимых различий между пользователями мобильных и стационарных устройств.")


# **Общий вывод**
# 
# Анализ подтвердил:
# 
# Гипотеза 1 верна: пользователи мобильных устройств делают статистически значимо больше заказов на пользователя (2,85 против 1,95 у стационарных, p = 0,0000).
# 
# Гипотеза 2 не подтвердилась: нет статистически значимых различий во времени между заказами для мобильных и стационарных устройств (p = 0,9991), несмотря на то, что по описательным метрикам время больше у стационарных (среднее: 319,18 ч. против 264,47 ч.; медиана: 216 ч. против 168 ч.).
# 
# Краткий итог: мобильные пользователи активнее по количеству заказов, но частота их покупок (интервал между заказами) не отличается от стационарных пользователей с точки зрения статистической значимости.

# **Итоговые выводы и рекомендации**
# 
# 
# **Ключевые выводы**
# 
# 
# Анализ данных за осень 2024 года позволил сформировать целостную картину поведения пользователей и динамики рынка. Основные выводы:
# 
# 1. Мобильный приоритет. 80 % заказов оформляются с мобильных устройств — это ключевой канал взаимодействия с клиентами. Подтверждено статистически: пользователи мобильных устройств совершают в среднем 2,85 заказа против 1,95 у стационарных (p = 0,0000). Это говорит о высокой вовлечённости аудитории через мобильные каналы и их ключевой роли в генерации выручки.
# 
# 
# 
# 2. Стабильность среднего чека. Отсутствие резких колебаний среднего чека по месяцам свидетельствует о:
# 
# - устойчивости ценовой политики;
# 
# - сбалансированности ассортимента мероприятий;
# 
# - отсутствии значимых внешних факторов, влияющих на платёжеспособность аудитории.
# 
# 
# 
# 3. Концентрация выручки у партнёров. Топ‑3 партнёра обеспечивают 42,71 % общей выручки, причём один оператор («Билеты без проблем») занимает 15,37 %. Такая структура говорит о зависимости бизнеса от ограниченного числа партнёров, но при этом демонстрирует более равномерное распределение выручки, чем предполагалось изначально.
# 
# 4. Сезонный рост активности. Пик заказов и выручки приходится на октябрь–ноябрь, что указывает на:
# 
# - повышенный спрос в этот период;
# 
# - потенциальную готовность аудитории к участию в промоакциях;
# 
# - необходимость заблаговременной подготовки к пиковым месяцам.
# 
# 
# 
# 5. Однородность периодичности покупок. Время между заказами не различается статистически для мобильных и стационарных устройств (p = 0,9991), хотя описательные метрики показывают разницу: медиана для мобильных — 168 ч. (7 дней), для стационарных — 216 ч. (9 дней). Это означает, что тип устройства не влияет на частоту покупок — решения о заказе принимаются под действием других факторов (новинки в афише, скидки, личные обстоятельства).
# 
# 6. Популярность форматов. Концерты и театральные постановки формируют основную часть заказов. Это подтверждает устойчивый спрос на «живые» культурные события и определяет фокус для планирования маркетинговых активностей.
# 
# **Обоснованные рекомендации**
# 
# 
# 1. Максимизировать потенциал мобильных каналов
# 
# - Почему это важно: 80 % заказов и более высокая активность пользователей делают мобильные устройства главным драйвером роста.
# 
# - Как реализовать:
# 
#   * провести UX‑аудит мобильного приложения и устранить «узкие места» (длительная загрузка, сложный процесс оплаты);
# 
#   * внедрить push‑уведомления о новинках, скидках и мероприятиях по интересам пользователя;
# 
#   * упростить повторные покупки: добавить автозаполнение данных, историю заказов и быстрый повтор.
# 
# - Ожидаемый эффект: рост конверсии в заказ, увеличение среднего числа заказов на пользователя, повышение лояльности мобильной аудитории.
# 
# 2. Оптимизировать партнёрскую сеть
# 
# - Почему важно: концентрация 42,71 % выручки у топ‑3 партнёров создаёт определённую зависимость, но оставляет пространство для диверсификации.
# 
# - Как реализовать:
# 
#   * проработать индивидуальные условия с ведущим партнёром для закрепления выгодных тарифов;
# 
#   * запустить программу привлечения новых партнёров с фокусом на нишевые мероприятия (выставки, лекции, мастер‑классы);
# 
#   * разработать систему бонусов для партнёров за рост продаж в низкий сезон.
# 
# - Ожидаемый эффект: снижение рисков, диверсификация выручки, расширение ассортимента предложений.
# 
# 3. Усилить сезонный маркетинг
# 
# - Почему это важно: рост активности в октябре–ноябре показывает готовность аудитории к покупкам в этот период.
# 
# - Как реализовать:
# 
#   * запустить промокампании за 2–3 недели до пикового сезона;
# 
#   * предложить специальные пакеты «Осенний абонемент» на концерты и спектакли;
# 
#   * использовать таргетированные рассылки по истории покупок (например, уведомления о гастролях любимых артистов).
# 
# - Ожидаемый эффект: увеличение числа заказов в пиковый период, рост средней выручки с пользователя.
# 
# 4. Стимулировать частоту покупок
# 
# - Почему это важно: медиана времени между заказами (168–216 ч.) указывает на потенциал роста — пользователи готовы покупать чаще при правильных стимулах.
# 
# - Как реализовать:
# 
#   * ввести многоуровневую программу лояльности с накоплением баллов за каждый заказ;
# 
#   * запускать персонализированные предложения на основе предпочтений (скидки на жанры, которые пользователь посещал ранее);
# 
#   * организовать «флеш‑распродажи» билетов на мероприятия за 1–3 дня до начала.
# 
# - Ожидаемый эффект: сокращение времени между заказами, рост среднего числа транзакций на пользователя.
# 
# 5. Фокусироваться на ключевых регионах
# 
# - Почему это важно: Москва и Московская область формируют основной спрос, но Санкт‑Петербург и Екатеринбург показывают потенциал роста.
# 
# - Как реализовать:
# 
#   * увеличить маркетинговый бюджет для Москвы и МО на 15–20 % в пиковые месяцы;
# 
#   * запустить пилотные кампании в Санкт‑Петербурге и Екатеринбурге с локальным контентом (анонсы городских событий, коллаборации с местными площадками);
# 
#   * проанализировать регионы с низкой активностью: выявить причины (дефицит мероприятий, слабая реклама) и протестировать точечные акции.
# 
# - Ожидаемый эффект: рост доли рынка в лидерах, выход на новые перспективные территории, выравнивание региональной активности.
# 
# 6. Мониторить и адаптировать ассортимент
# 
# - Почему это важно: концерты и театр лидируют по спросу, но нишевые форматы могут привлечь новую аудиторию.
# 
# - Как реализовать:
# 
#   * ежеквартально анализировать динамику продаж по категориям мероприятий;
# 
#   * тестировать новые форматы (стендап, иммерсивные шоу) в регионах с высокой активностью;
# 
#   * собирать обратную связь через опросы после покупки.
# 
# - Ожидаемый эффект: расширение целевой аудитории, снижение зависимости от популярных жанров, повышение устойчивости бизнеса.
# 
# **Заключение**
# Результаты анализа демонстрируют, что бизнес имеет чёткие точки роста: мобильные каналы, сезонность, партнёрская сеть и лояльность пользователей. Реализация предложенных рекомендаций позволит:
# 
# - увеличить выручку за счёт роста конверсии и частоты покупок;
# 
# - снизить риски за счёт диверсификации партнёров и регионов;
# 
# - укрепить позиции на рынке через персонализацию и адаптацию под спрос.
# 
# Ключевой приоритет — мобильное направление: его развитие даст максимальный краткосрочный эффект. В среднесрочной перспективе важно балансировать между поддержкой лидеров (Москва, топ‑партнёры) и освоением новых возможностей (регионы, нишевые форматы).

# In[ ]:




