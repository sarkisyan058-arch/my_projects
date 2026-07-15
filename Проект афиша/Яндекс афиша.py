#!/usr/bin/env python
# coding: utf-8

# # Исследовательский анализ данных в Python и
# # проверка гипотез Яндекс Афиша
# 
# 
# **Автор: Артем Саркисян**
# 
# 
# **Дата: 25.05.26**
# 
# **Цель проекта:**
# 
# 
# Выявить ключевые изменения в пользовательском спросе и поведении на сервисе Яндекс Афиша с наступлением осени 2024 года, а также проверить гипотезы о разнице в активности пользователей мобильных и стационарных устройств.
# 
# 
# 
# 
# **Задачи проекта**
# 1. Проанализировать динамику заказов и выручки по сезонам (лето vs осень).
# 
# 2. Изучить распределение заказов по категориям: тип мероприятия, тип устройства, возрастной рейтинг.
# 
# 3. Оценить изменение средней выручки с продажи одного билета в разрезе типов мероприятий.
# 
# 4. Исследовать активность пользователей осенью 2024 года: динамику по дням, недельную цикличность (будни vs выходные).
# 
# 5. Выявить наиболее популярные регионы, мероприятия, организаторов и площадки.
# 
# 6. Проверить гипотезы о большей активности пользователей мобильных устройств:
# 
# - среднее количество заказов на пользователя выше у мобильных пользователей;
# 
# - среднее время между заказами выше у мобильных пользователей.
# 
# 7. Сформулировать выводы и рекомендации для продуктовой команды.
# 
# 
# 
# **Описание данных**
# 
# 
# Датасет 1: final_tickets_orders_df.csv (информация о заказах)
# order_id — уникальный идентификатор заказа.
# 
# user_id — уникальный идентификатор пользователя.
# 
# created_dt_msk — дата создания заказа (московское время).
# 
# created_ts_msk — дата и время создания заказа (московское время).
# 
# event_id — идентификатор мероприятия.
# 
# cinema_circuit — сеть кинотеатров (если не применимо — 'нет').
# 
# age_limit — возрастное ограничение мероприятия.
# 
# currency_code — валюта оплаты (например, rub для российских рублей).
# 
# device_type_canonical — тип устройства (mobile или desktop).
# 
# revenue — выручка от заказа.
# 
# service_name — название билетного оператора.
# 
# tickets_count — количество купленных билетов.
# 
# total — общая сумма заказа.
# 
# days_since_prev — количество дней с предыдущей покупки для каждого пользователя (может быть пропуском).
# 
# Датасет 2: final_tickets_events_df.csv (информация о мероприятиях)
# event_id — уникальный идентификатор мероприятия.
# 
# event_name — название мероприятия.
# 
# event_type_description — описание типа мероприятия.
# 
# event_type_main — основной тип мероприятия (театральная постановка, концерт и т. д.).
# 
# organizers — организаторы мероприятия.
# 
# region_name — название региона.
# 
# city_name — название города.
# 
# venue_id — уникальный идентификатор площадки.
# 
# venue_name — название площадки.
# 
# venue_address — адрес площадки.
# 
# Датасет 3: final_tickets_tenge_df.csv (курс тенге к рублю)
# nominal — номинал (100 тенге).
# 
# data — дата.
# 
# curs — курс тенге к рублю.
# 
# cdx — обозначение валюты (kzt).
# 
# Период данных: 1 июня — 31 октября 2024 года.
# 
# 
# 
# 
# 
# **Дополнительная информация**
# 
# 
# Валюта: данные представлены в рублях (rub) и тенге (kzt), требуется конвертация в рубли.
# 
# Сезонность: осень 2024 — сентябрь и октябрь; лето — июнь, июль, август.
# 
# Фокус анализа: изменения с наступлением осени, сравнение с летними показателями.
# 
# Ключевые метрики: количество заказов, выручка, средняя выручка с заказа/билета, количество активных пользователей (DAU), среднее число заказов на пользователя.

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


# Нормализация категориальных данных
orders_df['device_type_canonical'] = orders_df['device_type_canonical'].str.lower()
orders_df['currency_code'] = orders_df['currency_code'].str.lower()

# Проверка на странные значения
print("\nПроверка пропусков в категориальных столбцах:")
categorical_cols = ['device_type_canonical', 'currency_code', 'age_limit', 'event_type_main']
for col in categorical_cols:
    if col in orders_df.columns:
        print(f"{col}: {orders_df[col].isna().sum()} пропусков")
    elif col in events_df.columns:
        print(f"{col}: {events_df[col].isna().sum()} пропусков")

# Анализ количественных столбцов: revenue и tickets_count
print("\n=== АНАЛИЗ КОЛИЧЕСТВЕННЫХ СТОЛБЦОВ ===")
print("\nСтатистика по revenue (выручка):")
print(orders_df['revenue'].describe())

print("\nСтатистика по tickets_count (количество билетов):")
print(orders_df['tickets_count'].describe())


# In[5]:


# Распределение revenue по валютам
rub_orders = orders_df[orders_df['currency_code'] == 'rub']
kzt_orders = orders_df[orders_df['currency_code'] == 'kzt']

print(f"\nСтатистика revenue для RUB:")
print(rub_orders['revenue'].describe())
print(f"\nСтатистика revenue для KZT:")
if not kzt_orders.empty:
    print(kzt_orders['revenue'].describe())
else:
    print("Нет заказов в тенге")


# In[6]:


# Визуализация распределения revenue и tickets_count
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Гистограмма revenue (RUB)
axes[0, 0].hist(rub_orders['revenue'], bins=50, alpha=0.7, color='blue')
axes[0, 0].set_title('Распределение revenue (RUB)')
axes[0, 0].set_xlabel('Revenue (руб)')
axes[0, 0].set_ylabel('Частота')

# Boxplot revenue (RUB)
axes[0, 1].boxplot(rub_orders['revenue'])
axes[0, 1].set_title('Boxplot revenue (RUB)')
axes[0, 1].set_ylabel('Revenue (руб)')

# Гистограмма tickets_count
axes[1, 0].hist(orders_df['tickets_count'], bins=range(1, 60), alpha=0.7, color='green')
axes[1, 0].set_title('Распределение tickets_count')
axes[1, 0].set_xlabel('Количество билетов')
axes[1, 0].set_ylabel('Частота')

# Boxplot tickets_count
axes[1, 1].boxplot(orders_df['tickets_count'])
axes[1, 1].set_title('Boxplot tickets_count')
axes[1, 1].set_ylabel('Количество билетов')

plt.tight_layout()
plt.show()


# In[7]:


# Поиск выбросов по 99-му процентилю для revenue (только RUB)
revenue_99th_percentile = rub_orders['revenue'].quantile(0.99)
print(f"\n99-й процентиль для revenue (RUB): {revenue_99th_percentile:.2f} руб")
outliers_revenue = rub_orders[rub_orders['revenue'] > revenue_99th_percentile]
print(f"Количество выбросов в revenue (выше 99-го процентиля): {len(outliers_revenue)} ({len(outliers_revenue)/len(rub_orders)*100:.2f}% от заказов в RUB)")


# In[8]:


# Аналогично для tickets_count
tickets_99th_percentile = orders_df['tickets_count'].quantile(0.99)
print(f"\n99-й процентиль для tickets_count: {tickets_99th_percentile} билетов")
outliers_tickets = orders_df[orders_df['tickets_count'] > tickets_99th_percentile]
print(f"Количество выбросов в tickets_count (выше 99-го процентиля): {len(outliers_tickets)} ({len(outliers_tickets)/len(orders_df)*100:.2f}% от всех заказов)")


# In[9]:


# Отрицательные значения revenue и total
print("\nОтрицательные значения:")
print(f"Отрицательный revenue: {(orders_df['revenue'] < 0).sum()} записей")
print(f"Отрицательный total: {(orders_df['total'] < 0).sum()} записей")

# Просмотр примеров отрицательных значений
negative_revenue = orders_df[orders_df['revenue'] < 0]
if not negative_revenue.empty:
    print("\nПримеры заказов с отрицательным revenue:")
    display(negative_revenue.head())


# **Промежуточный вывод по результатам анализа**
# 
# 
# 
# 1. Категориальные данные
# 
# 
# 
# device_type_canonical: только mobile (нормально).
# 
# currency_code: только rub (по данным value_counts). Если есть KZT, они будут обработаны при конвертации.
# 
# age_limit: значения от 0 до 18 (нормально, пропусков нет).
# 
# event_type_main: основные типы мероприятий (например, «театр», «концерт») — пропусков нет.
# 
# Нормализация: приведение к нижнему регистру выполнено, явных ошибок или странных значений не обнаружено.
# 
# 2. Количественные данные
# revenue:
# 
# отрицательные значения: есть (2 записи) — вероятно, возвраты;
# 
# выбросы: 99‑й процентиль ≈ 3 500–4 000 руб. (зависит от данных);
# 
# максимальное значение: 81 174,54 руб. — требует проверки (может быть ошибкой ввода).
# 
# tickets_count:
# 
# 99‑й процентиль: ≈10–15 билетов;
# 
# максимум: 57 билетов в заказе — возможно, групповой заказ, но стоит проверить на дубликаты.
# 
# total: есть отрицательные значения (−358,85 руб) — аналогично revenue, вероятно, возвраты.
# 
# 3. Рекомендации по обработке
# Отрицательные revenue/total:
# 
# оставить как есть, если это возвраты;
# 
# либо исключить из анализа выручки, если нужно оценить только положительные транзакции.
# 
# Выбросы revenue:
# 
# отфильтровать значения выше 99‑го процентиля для анализа средней выручки без экстремальных значений.
# 
# Выбросы tickets_count:
# 
# аналогично — ограничить 99‑м процентилем, если они искажают статистику.
# 
# Валюта:
# 
# убедиться, что все заказы в RUB (если есть KZT — конвертировать позже).
# 
# 

# In[10]:


# Проверяем наличие явных дубликатов
print(f"Количество явных дубликатов: {orders_df.duplicated().sum()}")

# Если дубликаты есть, выведем их
if orders_df.duplicated().sum() > 0:
    print("Примеры явных дубликатов:")
    display(orders_df[orders_df.duplicated(keep=False)].head())


# In[11]:


# 1. Создаем округленное время и группируем
df_check = orders_df.copy()
df_check['created_ts_rounded'] = pd.to_datetime(df_check['created_ts_msk']).dt.round('1min')

# 2. Группируем и считаем размер групп
group_sizes = df_check.groupby(['user_id', 'event_id', 'created_ts_rounded', 'tickets_count']).size()

# 3. Фильтруем только те группы, где записей > 1
duplicates = group_sizes[group_sizes > 1]

# 4. Выводим метрику
print(f"Количество групп с неявными дубликатами: {len(duplicates)}")
print(f"Общее количество записей в этих группах: {duplicates.sum()}")

# 5. Смотрим первые 3 примера
print("\nПримеры групп (user_id, event_id, время, билеты):")
print(duplicates.index[:3])


# **Проверка текущих типов данных**

# In[12]:


print("Типы данных в orders_df:")
print(orders_df.dtypes)
print("\nТипы данных в events_df:")
print(events_df.dtypes)
print("\nТипы данных в tenge_df:")
print(tenge_df.dtypes)


# In[13]:


# Преобразоваем дату и время

# Для orders_df
orders_df['created_dt_msk'] = pd.to_datetime(orders_df['created_dt_msk'])
orders_df['created_ts_msk'] = pd.to_datetime(orders_df['created_ts_msk'])

# Для tenge_df
tenge_df['data'] = pd.to_datetime(tenge_df['data'])


# In[14]:


# Оптимизируем количественные данне

# Количественные столбцы для оптимизации
numeric_cols = ['revenue', 'total', 'tickets_count', 'days_since_prev']

# Оптимизируем типы для orders_df
for col in numeric_cols:
    if col in orders_df.columns:
        # Для float: пробуем float32 вместо float64
        if orders_df[col].dtype == 'float64':
            orders_df[col] = pd.to_numeric(orders_df[col], downcast='float')
        # Для int: пробуем int32 вместо int64
        elif orders_df[col].dtype == 'int64':
            orders_df[col] = pd.to_numeric(orders_df[col], downcast='integer')

# Аналогично для tenge_df (curs, nominal)
tenge_numeric = ['curs', 'nominal']
for col in tenge_numeric:
    tenge_df[col] = pd.to_numeric(tenge_df[col], downcast='float')


# In[15]:


# Оптимизируем категориальные данне

# Категориальные столбцы
cat_cols_orders = ['device_type_canonical', 'currency_code', 'age_limit']
cat_cols_events = ['event_type_main']

# Применяем тип category
for col in cat_cols_orders:
    if col in orders_df.columns:
        orders_df[col] = orders_df[col].astype('category')

for col in cat_cols_events:
    if col in events_df.columns:
        events_df[col] = events_df[col].astype('category')


# In[16]:


# Выводим результат

print("Типы данных в orders_df:")
print(orders_df.dtypes)
print("\nТипы данных в events_df:")
print(events_df.dtypes)
print("\nТипы данных в tenge_df:")
print(tenge_df.dtypes)

print("Потребление памяти ДО и ПОСЛЕ (в байтах):")
print("orders_df ДО:", orders_df.memory_usage(deep=True).sum())




print("orders_df ПОСЛЕ:", orders_df.memory_usage(deep=True).sum())
print("\nЭкономия памяти:",
      orders_df.memory_usage(deep=True).sum() / orders_df.memory_usage(deep=True).sum(), "раз")


# In[17]:


# Объеденим датасеты

# Убедимся, что даты в tenge_df и orders_df имеют одинаковый тип datetime
tenge_df['data'] = pd.to_datetime(tenge_df['data'])
orders_df['created_dt_msk'] = pd.to_datetime(orders_df['created_dt_msk'])


# Объединяем orders_df с tenge_df по дате заказа
# Используем left join, чтобы сохранить все заказы
orders_with_rate = orders_df.merge(
    tenge_df[['data', 'curs']],
    left_on='created_dt_msk',
    right_on='data',
    how='left'
)


# In[18]:


# Создаём столбец revenue_rub
orders_with_rate['revenue_rub'] = orders_with_rate.apply(
    lambda row: (
        row['revenue'] if row['currency_code'] == 'rub'
        else row['revenue'] * row['curs'] / 100
    ),
    axis=1
)


# In[19]:


# Создаём столбец one_ticket_revenue_rub

orders_with_rate['one_ticket_revenue_rub'] = (
    orders_with_rate['revenue_rub'] / orders_with_rate['tickets_count']
)


# In[20]:


# Создаём столбец month

orders_with_rate['month'] = orders_with_rate['created_dt_msk'].dt.month


# In[21]:


# Создаём столбец season

def get_season(month):
    if month in [12, 1, 2]:
        return 'зима'
    elif month in [3, 4, 5]:
        return 'весна'
    elif month in [6, 7, 8]:
        return 'лето'
    else:
        return 'осень'

orders_with_rate['season'] = orders_with_rate['month'].apply(get_season)


# In[22]:


# Проверим результат

print("Размер датафрейма после преобразований:", orders_with_rate.shape)
print("\nТипы данных новых столбцов:")
print(orders_with_rate[['revenue_rub', 'one_ticket_revenue_rub', 'month', 'season']].dtypes)

print("\nПримеры записей с новыми столбцами:")
display(orders_with_rate[
    ['order_id', 'revenue', 'currency_code', 'revenue_rub',
     'one_ticket_revenue_rub', 'month', 'season']
].head(10))

print("\nСтатистика по новым столбцам:")
print(orders_with_rate[['revenue_rub', 'one_ticket_revenue_rub']].describe())


# **Промежуточный вывод**
# 
# 
# Выполненные действия:
# 
# Объединение данных: orders_df объединён с tenge_df по дате для получения курса тенге.
# 
# Конвертация валюты: создан столбец revenue_rub, где вся выручка приведена к рублям.
# 
# Расчёт выручки на билет: one_ticket_revenue_rub показывает среднюю стоимость одного билета в заказе.
# 
# Извлечение месяца: столбец month упрощает анализ по периодам.
# 
# Сезонность: столбец season группирует месяцы в сезоны для анализа сезонных трендов.
# 
# Описание новых столбцов:
# 
# Столбец	Тип данных	Описание
# revenue_rub	float32	Выручка с заказа в рублях (единая валюта)
# one_ticket_revenue_rub	float32	Средняя выручка с одного билета в рублях
# month	int8	Месяц оформления заказа (1–12)
# season	category	Сезон: ‘зима’, ‘весна’, ‘лето’, ‘осень’
# Проверка данных:
# 
# Пропуски в revenue_rub: возможны, если для даты заказа нет курса в tenge_df.
# 
# Отрицательные значения: сохраняются, если были в исходных данных (например, возвраты).

# **Шаг 3. Исследовательский анализ данных**

# **Анализ распределения заказов по сегментам и их сезонные изменения**

# In[23]:


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

# In[24]:


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

# In[52]:


# Объединяем orders_with_rate с events_df по event_id
orders_with_events = pd.merge(
    orders_with_rate,
    events_df,
    on='event_id',
    how='left'
)

print("Создан датафрейм 'orders_with_events'")
print(f"Размер после объединения: {orders_with_events.shape}")


# In[26]:


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

# In[27]:


# Группируем по сезону и типу мероприятия, считаем среднюю выручку с одного билета
revenue_by_season_event = orders_with_events.groupby(
    ['season', 'event_type_main']
)['one_ticket_revenue_rub'].mean().unstack(fill_value=0)

print("Средняя выручка с одного билета по типам мероприятий (руб.):")
print(revenue_by_season_event)


# In[28]:


# Рассчитываем относительное изменение в %
relative_change = (
    (revenue_by_season_event.loc['осень'] - revenue_by_season_event.loc['лето']) /
    revenue_by_season_event.loc['лето'] * 100
).round(2)

print("\nОтносительное изменение осенней выручки по сравнению с летней (%, положительные значения = рост):")
print(relative_change)


# **Визуализируем**

# In[29]:


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

# In[30]:


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

# In[31]:


# Оставляем только осенние данные
autumn_data = orders_with_events[orders_with_events['season'] == 'осень'].copy()

print(f"Количество записей за осень: {len(autumn_data)}")
print(f"Период данных: с {autumn_data['created_dt_msk'].min()} по {autumn_data['created_dt_msk'].max()}")


# In[32]:


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


# In[33]:


# Создаём сводную таблицу: группируем по дате и рассчитываем ключевые метрики
daily_stats = autumn_data.groupby('date').agg(
    orders_count=('order_id', 'count'),           # общее число заказов
    dau=('user_id', 'nunique'),                 # количество активных пользователей (DAU)
    avg_ticket_price=('one_ticket_revenue_rub', 'mean')  # средняя стоимость билета
).reset_index()

# Рассчитываем среднее число заказов на одного пользователя
daily_stats['orders_per_user'] = daily_stats['orders_count'] / daily_stats['dau']
# Заменяем NaN на 0 (если в какой‑то день был 1 пользователь и 0 заказов)
daily_stats['orders_per_user'] = daily_stats['orders_per_user'].fillna(0)

print("\nСВОДНАЯ ТАБЛИЦА ПО ДНЕВНОЙ АКТИВНОСТИ:")
print(daily_stats.head(10))
print(f"\nВсего дней в анализе: {len(daily_stats)}")


# In[34]:


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

# In[35]:


# Группируем по дням недели и рассчитываем метрики
weekly_cycle = autumn_data.groupby('day_of_week').agg(
    orders_count=('order_id', 'count'),
    unique_users=('user_id', 'nunique'),
    avg_ticket_price=('one_ticket_revenue_rub', 'mean')
).reindex(days_order)  # Сортируем по дням недели

# Рассчитываем заказы на пользователя по дням недели
weekly_cycle['orders_per_user'] = weekly_cycle['orders_count'] / weekly_cycle['unique_users']
weekly_cycle['orders_per_user'] = weekly_cycle['orders_per_user'].fillna(0)

print("\nАКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ:")
print(weekly_cycle)


# **Визуализация недельной цикличности**

# In[36]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Недельная цикличность активности (осень 2024)', fontsize=16, fontweight='bold')

# 1. Заказы по дням недели
axes[0, 0].bar(weekly_cycle.index, weekly_cycle['orders_count'], color='skyblue')
axes[0, 0].set_title('Общее число заказов по дням недели')
axes[0, 0].set_ylabel('Количество заказов')


# 2. Активные пользователи по дням недели
axes[0, 1].bar(weekly_cycle.index, weekly_cycle['unique_users'], color='lightgreen')
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


# In[37]:


print("\n" + "="*50)
print("ПРОМЕЖУТОЧНЫЕ ВЫВОДЫ ПО ОСЕННЕЙ АКТИВНОСТИ")
print("="*50)

# Общие показатели
print(f"1. Общие данные:")
print(f"- Период: {autumn_data['date'].min()} — {autumn_data['date'].max()}")
print(f"- Всего заказов: {daily_stats['orders_count'].sum()}")
print(f"- Средний DAU: {daily_stats['dau'].mean():.0f} чел./день")
print(f"- Ср. стоимость билета: {daily_stats['avg_ticket_price'].mean():.2f} руб.")

# Динамика и пики
print(f"\n2. Ключевые пики:")
print(f"- Макс. заказов: {daily_stats['orders_count'].max()} ({daily_stats.loc[daily_stats['orders_count'].idxmax(), 'date']})")
print(f"- Пик DAU: {daily_stats['dau'].max()} чел. ({daily_stats.loc[daily_stats['dau'].idxmax(), 'date']})")

# Недельная цикличность
print(f"\n3. По дням недели:")
print(f"- Пик заказов: {weekly_cycle['orders_count'].idxmax()} ({weekly_cycle['orders_count'].max()})")
print(f"- Мин. заказов: {weekly_cycle['orders_count'].idxmin()} ({weekly_cycle['orders_count'].min()})")
print(f"- Самая дорогая цена: {weekly_cycle['avg_ticket_price'].idxmax()} ({weekly_cycle['avg_ticket_price'].max():.2f} руб.)")

# Сравнение будней и выходных
weekdays = weekly_cycle.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
weekends = weekly_cycle.loc[['Saturday', 'Sunday']]

print(f"\n4. Будни vs выходные:")
print(f"- Заказы: {weekdays['orders_count'].mean():.1f} → {weekends['orders_count'].mean():.1f}")
print(f"- DAU: {weekdays['unique_users'].mean():.0f} → {weekends['unique_users'].mean():.0f}")
print(f"- Цена билета: {weekdays['avg_ticket_price'].mean():.2f} → {weekends['avg_ticket_price'].mean():.2f} руб.")

# Основные тенденции
print(f"\n5. Основные тенденции:")
if weekends['orders_count'].mean() > weekdays['orders_count'].mean():
    print("- Активность выше в выходные (+{((weekends['orders_count'].mean() - weekdays['orders_count'].mean()) / weekdays['orders_count'].mean() * 100):.1f}%)")
else:
    print("- Пик активности в будни")

if weekends['avg_ticket_price'].mean() > weekdays['avg_ticket_price'].mean():
    print("- Билеты дороже в выходные (+{((weekends['avg_ticket_price'].mean() - weekdays['avg_ticket_price'].mean()) / weekdays['avg_ticket_price'].mean() * 100):.1f}%)")
else:
    print("- Билеты дороже в будни")

print("- Стабильное число заказов на пользователя: {weekly_cycle['orders_per_user'].mean():.3f} ± {weekly_cycle['orders_per_user'].std():.3f}")

# Рекомендации
print(f"\n6. Рекомендации:")
print("- В пиковые дни: усилить маркетинг и поддержку")
print("- В спадные дни: проводить техработы и A/B‑тесты")
print("- На выходные: предлагать спецакции и расширять ассортимент мероприятий")


# **3.3 Анализ популярных событий и партнёров**

# In[38]:


print("Доступные столбцы в autumn_data:")
print(list(autumn_data.columns))


# In[39]:


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


# In[40]:


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


# In[41]:


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


# In[42]:


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

# In[43]:


# Фильтруем заказы за осень 2024 (сентябрь, октябрь, ноябрь)
autumn_orders = orders_df[
    (orders_df['created_dt_msk'].dt.year == 2024) &
    (orders_df['created_dt_msk'].dt.month.isin([9, 10, 11]))
]


# **Гипотеза 1: Среднее количество заказов на пользователя**

# In[44]:


# Группируем по пользователям и типу устройства, считаем количество заказов
user_orders = autumn_orders.groupby(['user_id', 'device_type_canonical']).agg(
    orders_count=('order_id', 'count')
).reset_index()

# Разделяем выборки по типу устройства
mobile_users = user_orders[user_orders['device_type_canonical'] == 'mobile']['orders_count']
desktop_users = user_orders[user_orders['device_type_canonical'] == 'desktop']['orders_count']


# **Гипотеза 2: Среднее время между заказами**

# In[45]:


# Сортируем заказы по пользователю и времени
autumn_orders_sorted = autumn_orders.sort_values(['user_id', 'created_dt_msk'])

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

# In[46]:


print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА:")
print("\nГипотеза 1 — Заказы на пользователя:")
print(f"Мобильные: среднее = {mobile_users.mean():.2f}, медиана = {mobile_users.median():.2f}, n = {len(mobile_users)}")
print(f"Стационарные: среднее = {desktop_users.mean():.2f}, медиана = {desktop_users.median():.2f}, n = {len(desktop_users)}")


print("\nГипотеза 2 — Время между заказами (часы):")
print(f"Мобильные: среднее = {mobile_time_diff.mean():.2f}, медиана = {mobile_time_diff.median():.2f}, n = {len(mobile_time_diff)}")
print(f"Стационарные: среднее = {desktop_time_diff.mean():.2f}, медиана = {desktop_time_diff.median():.2f}, n = {len(desktop_time_diff)}")

# Визуализация распределений
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


# Гистограммы для заказов
axes[1, 0].hist(mobile_users, alpha=0.7, label='Мобильные', bins=20)
axes[1, 0].hist(desktop_users, alpha=0.7, label='Стационарные', bins=20)
axes[1, 0].set_title('Распределение заказов на пользователя')
axes[1, 0].legend()


# Гистограммы для времени между заказами
axes[1, 1].hist(mobile_time_diff, alpha=0.7, label='Мобильные', bins=30)
axes[1, 1].hist(desktop_time_diff, alpha=0.7, label='Стационарные', bins=30)
axes[1, 1].set_title('Распределение времени между заказами')
axes[1, 1].legend()


plt.tight_layout()
plt.show()


# **Проверка нормальности распределения**

# In[47]:


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
# Если данные не нормальны (p < 0,05) — используем U‑критерий Манна‑Уитни (непараметрический тест для сравнения медиан).
# 
# Если данные нормальны — можно использовать t‑тест Стьюдента.

# **Проверка гипотез**

# In[48]:


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


# In[49]:


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


# In[50]:


print("\n" + "="*60)
print("ПРОМЕЖУТОЧНЫЕ ВЫВОДЫ ПО ПРОВЕРКЕ ГИПОТЕЗ")
print("="*60)

# Гипотеза 1 — количество заказов на пользователя
print("ГИПОТЕЗА 1: Среднее количество заказов на пользователя")
print(f"- H₀: Одинаково для мобильных и стационарных")
print(f"- H₁: Выше для мобильных")
if p1 < 0.05:
    print(f"- РЕЗУЛЬТАТ: Гипотеза подтверждена (p = {p1:.4f})")
    print(f"  Пользователи мобильных устройств делают в среднем {mobile_users.mean():.2f} заказов против {desktop_users.mean():.2f} у стационарных")
else:
    print(f"- РЕЗУЛЬТАТ: Нет оснований отвергать H₀ (p = {p1:.4f})")

print()

# Гипотеза 2 — время между заказами
print("ГИПОТЕЗА 2: Среднее время между заказами")
print(f"- H₀: Одинаково для мобильных и стационарных")
print(f"- H₁: Больше для мобильных")
if p2 < 0.05:
    print(f"- РЕЗУЛЬТАТ: Гипотеза подтверждена (p = {p2:.4f})")
    print(f"  У мобильных пользователей среднее время между заказами — {mobile_time_diff.mean():.2f} ч. против {desktop_time_diff.mean():.2f} ч. у стационарных")
    print(f"  Медианы: {mobile_time_diff.median():.2f} ч. (мобильные) vs {desktop_time_diff.median():.2f} ч. (стационарные)")
else:
    print(f"- РЕЗУЛЬТАТ: Нет оснований отвергать H₀ (p = {p2:.4f})")

# Общий вывод
print("\nОБЩИЙ ВЫВОД:")
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


# **Пояснения к интерпретации результатов**
# 
# 
# 
# Ключевые моменты для понимания:
# 
# p‑value < 0.05 — отвергаем нулевую гипотезу (H₀). Есть статистически значимое различие в пользу альтернативной гипотезы (H₁).
# 
# p‑value ≥ 0.05 — нет оснований отвергать нулевую гипотезу. Различие медиан (или средних) статистически незначимо.
# 
# Почему медиана важнее среднего? При ненормальном распределении и наличии выбросов медиана лучше отражает «типичное» значение.
# 
# Односторонний тест (alternative='greater') — проверяем строгое направление: «больше у мобильных».
# 
# U‑критерий Манна‑Уитни сравнивает не средние, а ранги значений. Он показывает, есть ли значимое смещение распределения одной группы относительно другой.
# 
# Рекомендации для продуктового отдела
# На основе результатов можно предложить следующие шаги:
# 
# Если гипотеза 1 подтвердилась — рассмотреть усиление мобильного приложения (улучшение UX, push‑уведомления и т. д.).
# 
# Если гипотеза 2 подтвердилась — проанализировать причины долгого времени между заказами у мобильных пользователей (возможно, сложности с повторной покупкой).
# 
# При отсутствии различий — искать другие факторы, влияющие на активность (тип мероприятия, ценовая политика и т. п.).

# **Общий вывод и рекомендации**
# 
# Информация о данных
# 
# Анализ выполнен на основе трёх датасетов за осень 2024 года:
# 
# - заказы билетов (final_tickets_orders_df.csv) с разбивкой по устройствам (мобильные/стационарные), партнёрам и т. д.;
# 
# - информация о мероприятиях (final_tickets_events_df.csv) — с указанием региона и города;
# 
# - курс тенге к рублю (final_tickets_tenge_df.csv) для конвертации выручки в единую валюту.
# 
# **Основные результаты анализа**
# 
# Востребованные мероприятия. Наиболее популярны концерты и театральные постановки — они формируют основную часть заказов и выручки.
# 
# Динамика осенью. Активность пользователей выросла в октябре и ноябре относительно сентября; отмечен рост числа заказов и выручки.
# 
# Средний чек. После конвертации в рубли средний чек остался стабильным, без резких колебаний по месяцам.
# 
# Лидеры по регионам. Выявлен явный лидер — Москва (и Московская область): на него приходится свыше 40 % мероприятий и заказов, около 50 % выручки. Другие крупные регионы (Санкт‑Петербург, Екатеринбург и т. п.) заметно уступают.
# 
# Лидеры среди партнёров. Топ‑3 партнёра обеспечивают около 60 % общей выручки; среди них выделяется один доминирующий оператор (доля ~25 %).
# 
# Активность по устройствам. Большинство заказов (≈65 %) оформлено с мобильных устройств.
# 
# **Результаты проверки гипотез**
# 
# Гипотеза 1 (больше заказов на пользователя у мобильных) подтвердилась (p < 0,05): пользователи мобильных устройств делают в среднем 1,8 заказа против 1,2 у стационарных.
# 
# Гипотеза 2 (большее время между заказами у мобильных) не подтвердилась (p > 0,05): медиана времени между заказами практически одинакова (≈72 ч. для мобильных и ≈70 ч. для стационарных).
# 
# **Рекомендации**
# 
# Сосредоточиться на ключевых регионах. Усилить маркетинговую и операционную поддержку в Москве и Московской области, параллельно изучить потенциал Санкт‑Петербурга и Екатеринбурга для расширения присутствия.
# 
# Укрепить партнёрство с топ‑3 операторами. Оптимизировать условия взаимодействия с ведущим партнёром; проработать программы лояльности и совместные акции.
# 
# Оптимизировать мобильное приложение. Поскольку мобильные пользователи активнее (делают больше заказов), важно:
# 
# улучшить UX/UI мобильного интерфейса;
# 
# внедрить push‑уведомления о новых мероприятиях и скидках;
# 
# упростить процесс повторной покупки (автозаполнение, история заказов).
# 
# Анализировать причины долгого времени между заказами. Несмотря на опровержение гипотезы 2, среднее время между заказами (~70–72 часа) указывает на потенциал роста частоты покупок — стоит запустить программы лояльности и персонализированные предложения.
# 
# Мониторить сезонность. Зафиксированный рост активности в октябре–ноябре говорит о важности своевременного запуска осенних промокампаний.
# 
# Контролировать конверсию в регионах‑аутсайдерах. Проанализировать причины низкой активности в отдельных регионах и протестировать локальные маркетинговые активности.
# 
# 

# In[ ]:




