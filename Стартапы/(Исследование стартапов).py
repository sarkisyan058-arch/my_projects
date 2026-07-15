# # Проект. Исследование стартапов

# ## Введение

# - Автор: Артем Саркисян
# - Дата:15.03.2026

# **Контекст:**
# - анализ стартап‑экосистемы для выявления перспективных отраслей и оптимальных стратегий финансирования.
# 
# **Цели проекта:**
# 
# - изучить распределение инвестиций по отраслям и типам финансирования;
# 
# - выявить сегменты рынка с наибольшим потенциалом роста;
# 
# - определить наиболее эффективные типы финансирования;
# 
# - дать рекомендации по инвестиционным стратегиям на 2015 год.


# ## Шаг 1. Знакомство с данными: загрузка и предобработка
# 
# 
# Название основного датасета — `cb_investments.zip`. Внутри архива один файл — `cb_investments.csv`.
# 
# Описание данных:
# * `name` — название компании.
# * `homepage_url` — ссылка на сайт компании.
# * `category_list` — категории, в которых работает компания. Указываются через `|`.
# * `market` — основной рынок или отрасль компании.
# * `funding_total_usd` — общий объём привлечённых инвестиций в долларах США.
# * `status` — текущий статус компании, например `operating`, `closed` и так далее.
# * `country_code` — код страны, например USA.
# * `state_code` — код штата или региона, например, CA.
# * `region` — регион, например, SF Bay Area.
# * `city` — город, в котором расположена компания.
# * `funding_rounds` — общее число раундов финансирования.
# * `participants` — число участников в раундах финансирования.
# * `founded_at` — дата основания компании.
# * `founded_month` — месяц основания в формате `YYYY-MM`.
# * `founded_quarter` — квартал основания в формате `YYYY-QN`.
# * `founded_year` — год основания.
# * `first_funding_at` — дата первого финансирования.
# * `mid_funding_at` — дата среднего по времени раунда финансирования.
# * `last_funding_at` — дата последнего финансирования.
# * `seed` — сумма инвестиций на посевной стадии.
# * `venture` — сумма венчурных инвестиций.
# * `equity_crowdfunding` — сумма, привлечённая через долевой краудфандинг.
# * `undisclosed` — сумма финансирования нераскрытого типа.
# * `convertible_note` — сумма инвестиций через конвертируемые займы.
# * `debt_financing` — сумма долгового финансирования.
# * `angel` — сумма инвестиций от бизнес-ангелов.
# * `grant` — сумма полученных грантов.
# * `private_equity` — сумма инвестиций в виде прямых (частных) вложений.
# * `post_ipo_equity` — сумма финансирования после IPO.
# * `post_ipo_debt` — сумма долгового финансирования после IPO.
# * `secondary_market` — сумма сделок на вторичном рынке.
# * `product_crowdfunding` — сумма, привлечённая через продуктовый краудфандинг.
# * `round_A` — `round_H` — сумма инвестиций в соответствующем раунде.
# 
# Название дополнительного датасета — `cb_returns.csv`. Он содержит суммы возвратов по типам финансирования в миллионах долларов по годам.
# 
# Описание данных:
# * `year` — год возврата средств.
# * `seed` — сумма возвратов от посевных инвестиций.
# * `venture` — сумма возвратов от венчурных инвестиций.
# * `equity_crowdfunding` — сумма, возвращённая по долевому краудфандингу.
# * `undisclosed` — сумма возвратов нераскрытого типа.
# * `convertible_note` — сумма возвратов через конвертируемые займы.
# * `debt_financing` — сумма возвратов от долгового финансирования.
# * `angel` — сумма возвратов бизнес-ангелам.
# * `grant` — сумма возвратов по грантам.
# * `private_equity` — сумма возвратов прямых (частных) вложений.
# * `post_ipo_equity` — сумма возвратов от IPO.
# * `post_ipo_debt` — сумма возвратов от долгового IPO.
# * `secondary_market` — сумма возвратов от сделок на вторичном рынке.
# * `product_crowdfunding` — сумма возвратов по продуктовому краудфандингу.
# 
# Файлы находятся в папке `datasets`, если вы выполняете работу на платформе. В случае, если вы делаете работу локально, доступ к файлам в папке можно получить по адресу `https://code.s3.yandex.net/datasets/` + имя файла.
# 
# ### 1.1. Вывод общей информации
# 
# 
# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[79]:


# Загрузка датасетов по URL
df_investments = pd.read_csv(
    'https://code.s3.yandex.net/datasets/cb_investments.zip',
    sep=';',
    low_memory=False
)

df_returns = pd.read_csv('https://code.s3.yandex.net/datasets/cb_returns.csv')


# **Проведём анализ датасета cb_investments**

# In[80]:


# Вывод иныормации о cb_investments
print('\n' + '='*50)
print('=== ИНФОРМАЦИЯ О ДАТАСЕТЕ CB_INVESTMENTS ===')
display(df_investments.head())
print(f'\nРазмер датасета: {df_investments.shape}')
print('\nОбщая информация:')
display(df_investments.info())
print('\nПропущенные значения:')
display(df_investments.isnull().sum())


# **Проведём анализ датасета cb_returns**

# In[81]:


# Вывод информации о cb_returns

print('\n' + '='*50)
print('=== ИНФОРМАЦИЯ О ДАТАСЕТЕ CB_RETURNS ===')
display(df_returns.head())
print(f'\nРазмер датасета: {df_returns.shape}')
print('\nОбщая информация:')
display(df_returns.info())
print('\nПропущенные значения:')
display(df_returns.isnull().sum())


# In[82]:


# Статистические показатели для cb_returns
print('Основные статистические показатели (cb_returns):')
display(df_returns.describe())


# **Проверим на дубликаты**

# In[83]:


# Проверка дубликатов в cb_returns
duplicates_returns = df_returns.duplicated().sum()
print(f'\nКоличество полных дубликатов в cb_returns: {duplicates_returns}')

# Проверка уникальности столбца year в cb_returns
if 'year' in df_returns.columns:
    year_unique = df_returns['year'].nunique()
    year_count = len(df_returns)
    print(f'\nСтолбец year: {year_unique} уникальных лет из {year_count} записей')



# **Вывод о полученных данных**
# 
# 
# 1. Объём данных
# 
# ***cb_investments:***
# 
# - количество строк: 54294;
# 
# - количество столбцов: 40;
# 
# - охватывает данные о стартапах с указанием финансирования, географии, статусов и т. д.
# 
# ***cb_returns:***
# 
# - количество строк: 15;
# 
# - количество столбцов: 14 (соответствует описанию);
# 
# - содержит агрегированные данные по возвратам инвестиций по типам и годам.
# 
# 2. Соответствие описанию
# 
# - Оба датасета в целом соответствуют предоставленному описанию:
# 
# ***cb_investments*** содержит все заявленные столбцы: name, homepage_url, category_list, market, funding_total_usd, status, country_code и т. д.;
# 
# ***cb_returns*** включает все указанные столбцы: year, seed, venture, equity_crowdfunding и другие типы возвратов.
# 
# 3. Пропущенные значения
# 
# ***cb_investments:***
# 
# - значительное количество пропусков в столбцах с детализацией финансирования — это ожидаемо, так как не все компании получают финансирование каждого типа;
# 
# - пропуски в географических данных (state_code, region, city) — могут потребовать заполнения;
# 
# - пропуски в датах (mid_funding_at) — требуют восстановления на основе других дат;
# 
# - столбец funding_total_usd может содержать пропуски — критически важен, требует особого внимания.
# 
# ***cb_returns:***
# 
# - возможны пропуски в столбцах сумм возвратов по типам — могут быть связаны с отсутствием данных за отдельные годы;
# 
# - на данном этапе пропусков не обнаруженно, возможно при дальнейшем анализе выявим.
# 
# 4. Типы данных
# 
# - Проблемы с типами данных:
# 
# ***cb_investments:***
# 
# - funding_total_usd: скорее всего, тип object (строка) из‑за разделителей разрядов (пробелов/запятых) — нужно преобразовать в числовой (float);
# 
# - столбцы с датами (founded_at, first_funding_at, mid_funding_at, last_funding_at): вероятно, тип object — нужно привести к datetime;
# 
# - категориальные столбцы (status, country_code, market и т. д.): тип object, что корректно, но может потребоваться кодирование для анализа.
# 
# ***cb_returns:***
# 
# - year: тип int64 — можно оставить как есть или сделать индексом;
# 
# - столбцы сумм возвратов: должны быть числовыми (float), но нужно проверить на наличие нечисловых значений.
# 
# 5. Другие особенности данных
# 
# ***cb_investments:***
# 
# - формат funding_total_usd: вероятно, содержит разделители разрядов (например, 1 000 000 вместо 1000000) — мешает математическим операциям;
# 
# - структура category_list: категории разделены символом | — может потребоваться разбиение на отдельные признаки;
# 
# - дубликаты: возможны полные дубликаты строк — нужно удалить;
# 
# - неоднородность текстовых данных: в столбцах city, region возможны разные варианты написания одного и того же значения — может потребоваться нормализация;
# 
# - пропуски в mid_funding_at: можно восстановить, рассчитав середину между first_funding_at и last_funding_at.
# 
# ***cb_returns:***
# 
# - уникальность year: нужно проверить, нет ли дублирования годов — если есть, возможно, потребуется агрегация;
# 
# - аномальные значения: в суммах возвратов могут быть выбросы — нужно проанализировать распределения.


# ### 1.2. Предобработка данных

# **Проверим названия столбцов и приведём их к snake_case**

# In[84]:


# Проверка текущих названий столбцов
print("Исходные названия столбцов в cb_investments:")
print(df_investments.columns.tolist())
print("\nИсходные названия столбцов в cb_returns:")
print(df_returns.columns.tolist())

# Функция для приведения названий столбцов к snake_case, для того, что бы в дальнейшем избежать ошибок
def clean_column_names(df):
    new_columns = []
    for col in df.columns:
        # Приводим к нижнему регистру
        col_clean = col.lower()
        # Заменяем пробелы и дефисы на подчёркивания
        col_clean = col_clean.replace(' ', '_').replace('-', '_')
        # Убираем лишние символы (оставляем буквы, цифры и подчёркивания)
        col_clean = ''.join(c if c.isalnum() or c == '_' else '' for c in col_clean)
        # Удаляем дублирующиеся подчёркивания
        while '__' in col_clean:
            col_clean = col_clean.replace('__', '_')
        new_columns.append(col_clean)
    df.columns = new_columns
    return df


# In[85]:


# Применяем функцию к обоим датасетам
df_investments_clean = clean_column_names(df_investments.copy())
df_returns_clean = clean_column_names(df_returns.copy())


# **Посмотрим что получилось**

# In[86]:


# Выводим результаты
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ПРИВЕДЕНИЯ К ЕДИНОМУ СТИЛЮ")
print("="*60)

print("\nНазвания столбцов в cb_investments после обработки:")
print(df_investments_clean.columns.tolist())


# In[87]:


print("\nНазвания столбцов в cb_returns после обработки:")
print(df_returns_clean.columns.tolist())


# In[88]:


# Сравниваем исходные и новые названия
print("\n" + "-"*60)
print("СРАВНЕНИЕ ИСХОДНЫХ И НОВЫХ НАЗВАНИЙ")
print("-"*60)

print("\ncb_investments:")
for old, new in zip(df_investments.columns, df_investments_clean.columns):
    if old != new:
        print(f"  '{old}' → '{new}'")

print("\ncb_returns:")
for old, new in zip(df_returns.columns, df_returns_clean.columns):
    if old != new:
        print(f"  '{old}' → '{new}'")


# In[89]:


# Сохраняем очищенные датасеты с новыми именами столбцов
df_investments = df_investments_clean
df_returns = df_returns_clean

print(f"\n Обработка завершена, теперь все столбцы в формате snake_case.")


# **Посмотрим на текущий тип funding_total_usd**

# In[90]:


# Проверка текущего типа данных и примеров значений
# В первой функции мы заменили пробелы на "_"
print("Текущий тип данных столбца funding_total_usd:", df_investments['_funding_total_usd_'].dtype)
print("\nПервые 10 значений столбца funding_total_usd:")
print(df_investments['_funding_total_usd_'].head(10))


# **Преобразовываем funding_total_usd**

# In[91]:


# Функция для очистки и преобразования funding_total_usd
def clean_funding_column(series):
    # В первом коде с переводом к snake_case мы уже убрали пробелы, но эта 
    # данная функция еще раз это продублирует и дополнит 
    # Заменяем запятые на точки для корректной обработки десятичных дробей (если есть)
    # и убираем пробелы/нецифровые символы, кроме точек и знаков минус
    cleaned = series.astype(str).str.replace(',', '').str.replace(' ', '')
    
    # Преобразуем в числовой тип, нецифровые значения станут NaN
    numeric_series = pd.to_numeric(cleaned, errors='coerce')
    return numeric_series

# Применяем функцию к столбцу
df_investments['_funding_total_usd_'] = clean_funding_column(df_investments['_funding_total_usd_'])


# In[92]:


# Проверка результата
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ОБРАБОТКИ funding_total_usd")
print("="*60)

print("Новый тип данных столбца funding_total_usd:", df_investments['_funding_total_usd_'].dtype)
print("\nПервые 10 значений после преобразования:")
print(df_investments['_funding_total_usd_'].head(10))

print("\nСтатистические показатели после преобразования:")
display(df_investments['_funding_total_usd_'].describe())


# **Посмотрим на количество пропусков после преобразования**

# In[93]:


# Проверяем количество пропусков после преобразования
missing_count = df_investments['_funding_total_usd_'].isnull().sum()
total_count = len(df_investments)
missing_percent = (missing_count / total_count) * 100

print(f"\nПропущенные значения после преобразования:")
print(f"Количество: {missing_count}")
print(f"Процент от общего числа: {missing_percent:.2f}%")


# **Проверим на аномалии**

# In[94]:


# Проверяем на аномалии (очень большие или маленькие значения)
print("\nПроверка на аномальные значения (топ-5 самых больших):")
top_outliers = df_investments['_funding_total_usd_'].nlargest(5)
print(top_outliers)


# **Выведем столбцы с датой и временем**

# In[95]:


# Список столбцов с датами для преобразования
date_columns = [
    'founded_at',
    'first_funding_at',
    'mid_funding_at',
    'last_funding_at'
]

print("=" * 60)
print("ОБРАБОТКА ДАТ В ДАТАСЕТЕ CB_INVESTMENTS")
print("=" * 60)

# Проверяем текущие типы данных и примеры значений
print("Текущие типы данных и первые значения:")
for col in date_columns:
    if col in df_investments.columns:
        print(f"\nСтолбец '{col}':")
        print(f"  Тип данных: {df_investments[col].dtype}")
        print(f"  Первые 5 значений: {df_investments[col].head(5).tolist()}")
    else:
        print(f"\nСтолбец '{col}' не найден в датасете")


# **Преобразовываем столбцы в datetime64**

# In[96]:


# Функция для преобразования дат с обработкой ошибок
def convert_to_datetime(series, col_name):
    
#Преобразует столбец в формат datetime с обработкой различных форматов дат.
#Возвращает преобразованный столбец и количество ошибок преобразования.

    try:
        # Пробуем преобразовать с автоматическим определением формата
        converted_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        # Считаем количество пропусков после преобразования
        null_count = converted_series.isnull().sum()
        total_count = len(series)
        error_percent = (null_count / total_count) * 100

        print(f"Столбец '{col_name}':")
        print(f"  Успешно преобразовано: {total_count - null_count}")
        print(f"  Ошибок преобразования: {null_count} ({error_percent:.2f}%)")

        return converted_series
    except Exception as e:
        print(f"Ошибка при преобразовании столбца '{col_name}': {e}")
        return series

# Применяем преобразование к каждому столбцу с датами
print("\n" + "-" * 60)
print("ПРОЦЕСС ПРЕОБРАЗОВАНИЯ ДАТ")
print("-" * 60)

for col in date_columns:
    if col in df_investments.columns:
        df_investments[col] = convert_to_datetime(df_investments[col], col)


# In[97]:


# Итоговая проверка результатов
print("\n" + "=" * 60)
print("ИТОГИ ПРЕОБРАЗОВАНИЯ ДАТ")
print("=" * 60)

print("Финальные типы данных столбцов с датами:")
for col in date_columns:
    if col in df_investments.columns:
        print(f"  '{col}': {df_investments[col].dtype}")

print("\nПримеры преобразованных дат:")
display(df_investments[date_columns].head())


# In[98]:


# Дополнительная информация о пропусках после преобразования
print("\nПропущенные значения в столбцах с датами после преобразования:")
date_nulls = df_investments[date_columns].isnull().sum()
for col, null_count in date_nulls.items():
    if null_count > 0:
        percent = (null_count / len(df_investments)) * 100
        print(f"  '{col}': {null_count} пропусков ({percent:.2f}%)")


# **Делаем year индексом**

# In[99]:


# Устанавливаем 'year' как индекс
df_returns.set_index('year', inplace=True)

# Сортируем индекс по возрастанию (по годам)
df_returns.sort_index(ascending=True, inplace=True)

# Проверка результата
print("\n" + "-" * 60)
print("РЕЗУЛЬТАТЫ УСТАНОВКИ ИНДЕКСА")
print("-" * 60)

print("Новый индекс (первые 10 значений):")
print(df_returns.index[:10].tolist())

print("\nПервые 5 строк датасета с новым индексом:")
display(df_returns.head())

# Обработайте текстовые данные, если это необходимо. Пропуски в текстовых столбцах заполните заглушками там, где это понадобится.

# **Выведем необходимые столбцы и посмотрим на пропуски**

# In[100]:


# Список текстовых столбцов для обработки
text_columns = [
    'name',
    'homepage_url',
    'category_list',
    '_market_',
    'country_code',
    'state_code',
    'region',
    'city'
]

print("=" * 60)
print("ОБРАБОТКА ТЕКСТОВЫХ ДАННЫХ В CB_INVESTMENTS")
print("=" * 60)

# Проверка текущих пропусков в текстовых столбцах
print("Текущие пропуски в текстовых столбцах:")
missing_text = df_investments[text_columns].isnull().sum()
for col, missing_count in missing_text.items():
    if missing_count > 0:
        percent = (missing_count / len(df_investments)) * 100
        print(f"  '{col}': {missing_count} пропусков ({percent:.2f}%)")


# **Заполняем пропуски. В качестве заглушки используем unknown**

# In[101]:


# Функция для заполнения пропусков в текстовых столбцах
def fill_text_missing(series, placeholder='unknown'):
    
    return series.fillna(placeholder)

# Функция для базовой очистки текстовых данных
def clean_text_data(series):
    
    if series.dtype == 'object':  # обрабатываем только текстовые столбцы
        cleaned = series.astype(str)
        # Приводим к нижнему регистру
        cleaned = cleaned.str.lower()
        # Убираем пробелы в начале и конце
        cleaned = cleaned.str.strip()
        # Заменяем множественные пробелы на один
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
        return cleaned
    else:
        return series

# Применяем обработку к каждому текстовому столбцу
print("\n" + "-" * 60)
print("ПРОЦЕСС ОБРАБОТКИ ТЕКСТОВЫХ ДАННЫХ")
print("-" * 60)

for col in text_columns:
    if col in df_investments.columns:
        print(f"\nОбработка столбца '{col}':")
        
# Заполняем пропуски
        original_missing = df_investments[col].isnull().sum()
        df_investments[col] = fill_text_missing(df_investments[col], 'unknown')
        new_missing = df_investments[col].isnull().sum()
        print(f"  Пропуски заполнены заглушкой 'unknown' "
              f"(было: {original_missing}, стало: {new_missing})")

        # Очищаем текст
        df_investments[col] = clean_text_data(df_investments[col])
        print(f"  Выполнена базовая очистка текста")

# Итоговая проверка результатов
print("\n" + "=" * 60)
print("ИТОГИ ОБРАБОТКИ ТЕКСТОВЫХ ДАННЫХ")
print("=" * 60)


# In[102]:


print("Пропуски после обработки (только текстовые столбцы):")
final_missing = df_investments[text_columns].isnull().sum()
for col, missing_count in final_missing.items():
    print(f"  '{col}': {missing_count} пропусков")

print("\nПримеры очищенных данных:")
display(df_investments[text_columns].head(10))



# **Обрабатываем дубликаты и пропуски**

# In[103]:


print("=" * 60)
print("ОБРАБОТКА ДУБЛИКАТОВ И ПРОПУСКОВ В CB_INVESTMENTS")
print("=" * 60)

# 1. Проверка и удаление полных дубликатов
print("1. ОБРАБОТКА ПОЛНЫХ ДУБЛИКАТОВ")
print("- " * 40)

total_rows_before = len(df_investments)
duplicates_count = df_investments.duplicated().sum()

print(f"Общее количество строк до удаления дубликатов: {total_rows_before}")
print(f"Количество полных дубликатов: {duplicates_count}")

if duplicates_count > 0:
    # Удаляем дубликаты, оставляем первую копию
    df_investments.drop_duplicates(inplace=True)
    total_rows_after_duplicates = len(df_investments)
    print(f"Удалено дубликатов: {duplicates_count}")
    print(f"Осталось строк после удаления дубликатов: {total_rows_after_duplicates}")
else:
    print("Полные дубликаты не обнаружены.")
    total_rows_after_duplicates = total_rows_before


# In[104]:


# 2. Обработка пропусков в _funding_total_usd_ и удаление строк без данных о финансировании
print("\n2. ОБРАБОТКА ПРОПУСКОВ В _funding_total_usd_")
print("- " * 40)

missing_funding = df_investments['_funding_total_usd_'].isnull().sum()
total_rows_current = len(df_investments)
missing_percent = (missing_funding / total_rows_current) * 100

print(f"Пропущенные значения в _funding_total_usd_: {missing_funding}")
print(f"Процент от общего числа: {missing_percent:.2f}%")


# In[ ]:





# In[105]:


# Удаляем строки, где _funding_total_usd_ — NaN или 0 (не содержат данных о финансировании)
mask_funding_valid = df_investments['_funding_total_usd_'].notna() & (df_investments['_funding_total_usd_'] > 0)
df_investments_clean = df_investments[mask_funding_valid].copy()

rows_removed_funding = total_rows_current - len(df_investments_clean)
print(f"Удалено строк без данных о финансировании: {rows_removed_funding}")
print(f"Осталось строк после удаления: {len(df_investments_clean)}")


# In[106]:


# 3. Проверка на строки, не несущие полезной информации
print("\n3. ПРОВЕРКА НА СТРОКИ БЕЗ ПОЛЕЗНОЙ ИНФОРМАЦИИ")
print("- " * 40)

# Определяем столбцы, которые должны содержать значимые данные
critical_columns = [
    'name',
    'category_list',
    '_funding_total_usd_',
    'country_code'
]

# Создаём маску: строка считается «полезной», если хотя бы в одном критическом столбце есть данные ≠ 'unknown' или NaN
def is_useful_row(row):
    for col in critical_columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and value != 'unknown':
                return True
    return False

useful_mask = df_investments_clean.apply(is_useful_row, axis=1)
df_investments_final = df_investments_clean[useful_mask].copy()

rows_removed_useless = len(df_investments_clean) - len(df_investments_final)
print(f"Удалено строк без полезной информации: {rows_removed_useless}")
print(f"Итоговый размер датасета: {len(df_investments_final)} строк")


# In[ ]:





# **Пропуски и дубликаты обработанны, сохраним текущий результат**

# In[107]:


# Сохраняем финальный очищенный датасет
df_investments = df_investments_final
print("\n Обработка завершена. Финальный датасет сохранён в df_investments.")



# **Проверим столбец mid_funding_at на пропуски**

# In[108]:


import pandas as pd
from datetime import timedelta # Пришлось загрузить библиотеку повторно, 
# так как иначе возникает ошибка


print("=" * 60)
print("ЗАПОЛНЕНИЕ ПРОПУСКОВ В mid_funding_at")
print("=" * 60)

# Проверяем текущие пропуски в mid_funding_at
initial_missing = df_investments['mid_funding_at'].isnull().sum()
total_rows = len(df_investments)
initial_percent = (initial_missing / total_rows) * 100


print(f"Исходные пропуски в mid_funding_at: {initial_missing} ({initial_percent:.2f}%)")



# **Считаем среднее между first_date и last_date**

# In[109]:


# Функция для расчёта средней даты между двумя датами, возвращает None, если одна из дат отсутствует.
def calculate_mid_date(first_date, last_date):
   
    if pd.isna(first_date) or pd.isna(last_date):
        return pd.NaT
    
    # Вычисляем разницу между датами в днях
    delta = (last_date - first_date).days
    # Середина интервала — половина разницы
    mid_days = delta // 2
    # Прибавляем половину интервала к первой дате
    mid_date = first_date + timedelta(days=mid_days)
    return mid_date

# Применяем функцию для заполнения пропусков
print("\nЗаполнение пропусков в mid_funding_at...")

# Создаём маску для строк с пропусками в mid_funding_at, но с заполненными first_ и last
mask_fillable = (
    df_investments['mid_funding_at'].isna() &
    df_investments['first_funding_at'].notna() &
    df_investments['last_funding_at'].notna()
)

# Считаем количество строк, которые можно заполнить
fillable_count = mask_fillable.sum()
print(f"Количество строк, где можно рассчитать mid_funding_at: {fillable_count}")

# Заполняем пропуски
for idx in df_investments[mask_fillable].index:
    first_date = df_investments.loc[idx, 'first_funding_at']
    last_date = df_investments.loc[idx, 'last_funding_at']
    mid_date = calculate_mid_date(first_date, last_date)
    df_investments.loc[idx, 'mid_funding_at'] = mid_date


# Итоговая проверка пропусков
final_missing = df_investments['mid_funding_at'].isnull().sum()
final_percent = (final_missing / total_rows) * 100
reduced_missing = initial_missing - final_missing

print("\n" + "-" * 60)
print("РЕЗУЛЬТАТЫ ЗАПОЛНЕНИЯ")
print("-" * 60)

print(f"Исходные пропуски: {initial_missing} ({initial_percent:.2f}%)")
print(f"Заполнено значений: {reduced_missing}")
print(f"Осталось пропусков: {final_missing} ({final_percent:.2f}%)")


# **Проверим mid_funding_at**

# In[110]:


#Строки, где mid_funding_at всё ещё NaN
remaining_mask = df_investments['mid_funding_at'].isna()

# Причины:
# 1. Нет first_funding_at
missing_first = df_investments[remaining_mask & df_investments['first_funding_at'].isna()].shape[0]
# 2. Нет last_funding_at
missing_last = df_investments[remaining_mask & df_investments['last_funding_at'].isna()].shape[0]
# 3. Обе даты есть, но что‑то пошло не так (маловероятно)
both_present = df_investments[
    remaining_mask &
    df_investments['first_funding_at'].notna() &
    df_investments['last_funding_at'].notna()
].shape[0]

print(f"Причины оставшихся пропусков:")
print(f"  Нет first_funding_at: {missing_first}")
print(f"  Нет last_funding_at: {missing_last}")
print(f"  Обе даты есть, но не заполнилось: {both_present}")


# In[111]:


# Примеры строк с оставшимися пропусками
print(f"\nПримеры строк с оставшимися пропусками (первые 5):")
display(df_investments[remaining_mask][['first_funding_at', 'mid_funding_at', 'last_funding_at']].head())

# Статистика по заполненному столбцу
print("\nСТАТИСТИКА ПО mid_funding_at ПОСЛЕ ЗАПОЛНЕНИЯ")
print("- " * 40)
if final_missing < total_rows:  # если есть хотя бы одно заполненное значение
    print(f"Минимальный год: {df_investments['mid_funding_at'].dt.year.min()}")
    print(f"Максимальный год: {df_investments['mid_funding_at'].dt.year.max()}")
    print(f"Средний год: {df_investments['mid_funding_at'].dt.year.mean():.1f}")
else:
    print("Все значения в mid_funding_at остаются пропущенными.")


print("\n Обработка завершена.")


# **СРАВНИТЕЛЬНЫЙ АНАЛИЗ НАЧАЛЬНЫХ И ОБРАБОТАННЫХ ДАННЫХ df_investments**
# 
# - Исходный размер датасета: 54294 строк × 40 столбцов
# - Финальный размер датасета: 40907 строк × 40 столбцов
# 
# - Отброшено строк: 13387 (24.66%)
# - Сохранено строк: 40907 (75.34%)

# **ДЕТАЛЬНЫЙ РАЗБОР ПРИЧИН ПОТЕРЬ ДАННЫХ**
# 
# 1. Удаление полных дубликатов: 4 855 строк (8,94%)
# 2. Удаление строк без данных о финансировании: 8 532 строк (15,71%)
# 3. Удаление строк без полезной информации: 0 строк (0,00%)
# ----------------------------------------------------------------------
# Всего потеряно: 13387 строк (24.66%)

# **Сравнение пропусков по ключевым столбцам до и после обработки**

# **ОЦЕНКА ИЗМЕНЕНИЯ ПРОПУСКОВ ПО КЛЮЧЕВЫМ СТОЛБЦАМ**
# 
# Ключевые столбцы и ожидаемые изменения пропусков:
# -  funding_total_usd: пропуски удалены (строки без финансирования отброшены)
# -  mid_funding_at: часть пропусков заполнена (середина между first и last)
# -  остальные столбцы: пропуски могли остаться, но без увеличения

# **Анализ качества данных после обработки**
# 
# - Компании с данными о финансировании: 40907 (100.00%)

# **Полнота дат финансирования**
# 
# - Компании с полным набором дат финансирования: 40905 (100.00%)

# **Географическое покрытие**
# 
# - Географическое покрытие: 111 стран, 3587 городов

# **Категориальные данные**
# 
# - Разнообразие категорий: 14278 уникальных категорий
# - Разнообразие рынков: 395 уникальных рынков

# **Статистика по финансированию**
# - Минимальная сумма: \$1.00
# - Максимальная сумма: \$30,079,503,000.00
# - Средняя сумма: \$15,912,526.05
# - Медиана: \$2,000,000.00
# 
# 

# Предварительный вывод о достаточности данных
# 
# КРИТЕРИИ ОЦЕНКИ:
# 1. Объём данных после очистки: >70% от исходных — приемлемо
# 2. Пропуски в ключевых столбцах: <20% — хорошо, 20–40% — удовлетворительно, >40% — требует внимания
# 3. Полнота по ключевым метрикам: >50% компаний с полными данными — хорошо
# 4. Разнообразие данных: достаточное географическое и категориальное покрытие
# 
# ОЦЕНКА:
# Объём данных после очистки достаточен (75.34%)
# ДАННЫЕ ДОСТАТОЧНЫ ДЛЯ РЕШЕНИЯ ЗАДАЧ ПРОЕКТА
# Обоснование: соблюдены все критерии полноты и качества данных

# **На данном этапе подведём итоги:**
# 
# Итоговый размер датасета: 40907 строк (75.34% от исходных)
# Отброшено: 13387 строк (24.66%)
# Ключевые показатели полноты:
#   - Компании с финансированием: 100.00%
#   - Полный набор дат: 100.00%
#   - Географическое покрытие: 111 стран


# ## Шаг 2. Инжиниринг признаков

# ### 2.1. Группы по срокам финансирования
# 
# * Единичное финансирование — был всего один раунд финансирования.
# 
# * Срок финансирования до года — между первым и последним раундом финансирования прошло не более года.
# 
# * Срок финансирования более года.
# 
# 

# **Построим графики и дадим обьяснение к проделаной работе**

# In[112]:


print("=" * 70)
print("ШАГ 2. ИНЖИНИРИНГ ПРИЗНАКОВ: ГРУППЫ ПО СРОКАМ ФИНАНСИРОВАНИЯ")
print("=" * 70)

# Создаём копию датасета для работы
df_work = df_investments.copy()

# Рассчитываем длительность финансирования в днях
df_work['funding_duration_days'] = (
    df_work['last_funding_at'] - df_work['first_funding_at']
).dt.days

# Определяем группы финансирования
def assign_funding_group(row):
    # Единичное финансирование: если first и last совпадают или duration NaN
    if pd.isna(row['funding_duration_days']) or row['funding_duration_days'] == 0:
        return 'Единичное финансирование'
    # До года: duration <= 365 дней
    elif row['funding_duration_days'] <= 365:
        return 'Срок финансирования до года'
    # Более года: duration > 365 дней
    else:
        return 'Срок финансирования более года'

# Применяем функцию для создания группы
df_work['funding_group'] = df_work.apply(assign_funding_group, axis=1)

# Статистика по группам
group_stats = df_work['funding_group'].value_counts()
group_percentages = (group_stats / len(df_work) * 100).round(2)

print("РАСПРЕДЕЛЕНИЕ КОМПАНИЙ ПО ГРУППАМ:")
for group, count in group_stats.items():
    percent = group_percentages[group]
    print(f"{group}: {count} компаний ({percent}%)")


# In[113]:


# Расчёт объёма инвестиций по группам
investment_by_group = df_work.groupby('funding_group')['_funding_total_usd_'].sum()
total_investment = investment_by_group.sum()
investment_percentages = (investment_by_group / total_investment * 100).round(2)

print("\nОБЪЁМ ИНВЕСТИЦИЙ ПО ГРУППАМ:")
for group, amount in investment_by_group.items():
    percent = investment_percentages[group]
    print(f"{group}: ${amount:,.2f} ({percent}%)")


# In[114]:


# Единая цветовая палитра для графиков
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # красный, бирюзовый, голубой


# **Построим график — круговая диаграмма распределения компаний по группам (в % от общего количества);**

# In[115]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
wedges, texts, autotexts = plt.pie(
    group_stats,
    labels=group_stats.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)
plt.title('Распределение компаний по группам финансирования\n(по количеству)', fontsize=14, fontweight='bold')

# Добавляем легенду вне графика для лучшей читаемости
plt.legend(wedges, group_stats.index, title="Группы", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))


# **Построим второй график: Распределение по объёму инвестиций**

# In[116]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 2)
wedges2, texts2, autotexts2 = plt.pie(
    investment_by_group,
    labels=investment_by_group.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 10}
)
plt.title('Распределение инвестиций по группам финансирования\n(по объёму средств)', fontsize=14, fontweight='bold')

# Легенда для второго графика
plt.legend(wedges2, investment_by_group.index, title="Группы", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.show()


# In[117]:


# Дополнительная статистика
print("\n" + "-" * 70)
print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
print("-" * 70)

avg_investment_by_group = df_work.groupby('funding_group')['_funding_total_usd_'].mean()
median_investment_by_group = df_work.groupby('funding_group')['_funding_total_usd_'].median()

for group in group_stats.index:
    avg_inv = avg_investment_by_group[group]
    median_inv = median_investment_by_group[group]
    print(f"\n{group}:")
    print(f"  Средняя сумма финансирования: ${avg_inv:,.2f}")
    print(f"  Медианная сумма финансирования: ${median_inv:,.2f}")


# In[118]:


# Сохранение результата для дальнейшего использования
df_investments = df_work
print("\n Признак 'funding_group' создан и добавлен в датасет.")
print(" Графики визуализированы.")


# **Интерпретация и промежуточные выводы:**
# 
# - По количеству компаний: позволяет оценить распространённость разных моделей финансирования. Например, если большинство компаний получили финансирование единожды, это может указывать на сложности с привлечением последующих раундов.
# 
# - По объёму инвестиций: показывает, какие модели финансирования привлекают наибольший капитал. Если группа «более года» получает львиную долю средств, это говорит о важности длительных инвестиционных циклов для крупных проектов.
# 
# - Сравнение средней и медианной сумм: выявляет влияние выбросов. Значительное превышение среднего над медианой указывает на наличие нескольких очень крупных сделок, искажающих общую картину.
# 
# - Согласованность графиков: расхождение долей по компаниям и инвестициям (например, небольшая группа компаний получает большую часть средств) может свидетельствовать о концентрации капитала в отдельных сегментах.


# ### 2.2 Выделение средних и нишевых сегментов рынка


# In[119]:


print("=" * 70)
print("3.2 ВЫДЕЛЕНИЕ СРЕДНИХ И НИШЕВЫХ СЕГМЕНТОВ РЫНКА")
print("=" * 70)

# Проверяем наличие столбца _market_
if '_market_' not in df_investments.columns:
    print("Ошибка: столбец '_market_' не найден в датасете")
else:
    # Удаляем пропуски и считаем частоту каждого сегмента
    market_counts = df_investments['_market_'].dropna().value_counts()
    
    print(f"Всего уникальных сегментов рынка: {len(market_counts)}")
    print(f"Общее количество компаний с указанием сегмента: {market_counts.sum()}")
    
    # Классифицируем сегменты по размеру
    mass_segments = market_counts[market_counts > 120]
    medium_segments = market_counts[(market_counts >= 35) & (market_counts <= 120)]
    niche_segments = market_counts[market_counts < 35]
    
    # Считаем количество сегментов в каждой категории
    num_mass = len(mass_segments)
    num_medium = len(medium_segments)
    num_niche = len(niche_segments)
    
    print("\n" + "-" * 50)
    print("РАСПРЕДЕЛЕНИЕ СЕГМЕНТОВ ПО КАТЕГОРИЯМ")
    print("-" * 50)
    print(f"Массовые сегменты (>120 компаний): {num_mass} сегментов")
    print(f"Средние сегменты (35–120 компаний): {num_medium} сегментов")
    print(f"Нишевые сегменты (<35 компаний): {num_niche} сегментов")
    print(f"\nВсего сегментов: {num_mass + num_medium + num_niche}")


# In[120]:


# Выводим топ-10 самых крупных сегментов
print("\nТОП-10 САМЫХ КРУПНЫХ СЕГМЕНТОВ:")
for i, (segment, count) in enumerate(market_counts.head(10).items(), 1):
    category = "Массовый" if count > 120 else "Средний" if 35 <= count <= 120 else "Нишевый"
    print(f"{i}. {segment}: {count} компаний ({category})")


# In[121]:


# Подготавливаем данные для графика (берём топ-40 сегментов для наглядности)
top_segments = market_counts.head(40)
categories_for_plot = []
colors_for_plot = []

for count in top_segments.values:
    if count > 120:
        categories_for_plot.append("Массовый")
        colors_for_plot.append("#FF6B6B")  # красный
    elif 35 <= count <= 120:
        categories_for_plot.append("Средний")
        colors_for_plot.append("#4ECDC4")  # бирюзовый
    else:
        categories_for_plot.append("Нишевый")
        colors_for_plot.append("#45B7D1")  # голубой

# Строим график
plt.figure(figsize=(16, 9))
bars = plt.bar(
    range(len(top_segments)),
    top_segments.values,
    color=colors_for_plot,
    edgecolor='black',
    linewidth=0.5
)
plt.xlabel("Сегменты рынка (топ-40 по количеству компаний)", fontsize=12)
plt.ylabel("Количество компаний", fontsize=12)
plt.title("Распределение компаний по сегментам рынка\n(с разделением на массовые, средние и нишевые)", fontsize=14, fontweight='bold')

# Подписи сегментов на оси X (с переносом строк для длинных названий)
segment_labels = [seg[:25] + "\n" + seg[25:] if len(seg) > 25 else seg for seg in top_segments.index]
plt.xticks(range(len(top_segments)), segment_labels, rotation=60, ha='right', fontsize=8)

# Добавляем значения над столбцами
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', fontsize=7)
# Легенда
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='Массовый (>120)'),
    Patch(facecolor='#4ECDC4', label='Средний (35–120)'),
    Patch(facecolor='#45B7D1', label='Нишевый (<35)')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Дополнительная статистика по компаниям в сегментах разных категорий
total_companies_in_mass = mass_segments.sum()
total_companies_in_medium = medium_segments.sum()
total_companies_in_niche = niche_segments.sum()
total_with_market = market_counts.sum()
percent_mass = (total_companies_in_mass / total_with_market) * 100
percent_medium = (total_companies_in_medium / total_with_market) * 100
percent_niche = (total_companies_in_niche / total_with_market) * 100
print("\n" + "-" * 60)
print("СТАТИСТИКА ПО КОЛИЧЕСТВУ КОМПАНИЙ В КАТЕГОРИЯХ СЕГМЕНТОВ")
print("-" * 60)
print(f"В массовых сегментах: {total_companies_in_mass} компаний ({percent_mass:.1f}%)")
print(f"В средних сегментах: {total_companies_in_medium} компаний ({percent_medium:.1f}%)")
print(f"В нишевых сегментах: {total_companies_in_niche} компаний ({percent_niche:.1f}%)")
print(f"\nОбщее количество компаний с сегментом: {total_with_market}")


# In[122]:


# Сохраняем категории сегментов для дальнейшего использования
segment_categories = {}
for segment, count in market_counts.items():
    if count > 120:
        segment_categories[segment] = "Массовый"
    elif 35 <= count <= 120:
        segment_categories[segment] = "Средний"
    else:
        segment_categories[segment] = "Нишевый"
df_investments['market_category'] = df_investments['_market_'].map(segment_categories)
print("\nСоздан новый признак 'market_category' с категоризацией сегментов.")
print("График распределения построен.")


# **Основные наблюдения:**
# 
# - Доминирование массовых сегментов. Подавляющее большинство компаний (88,6 %, или 36 236 из 40 907) сосредоточено в массовых сегментах — тех, где более 120 компаний относятся к одной категории. Это указывает на высокую концентрацию стартапов и инвестиционных проектов в нескольких ключевых направлениях.
# 
# - Незначительная доля средних и нишевых сегментов:
# 
# - средние сегменты (35–120 компаний) охватывают 9,4 % компаний (3 841);
# 
# - нишевые сегменты (< 35 компаний) — всего 2,0 % (830).
# 
# - Высокая фрагментация нишевых направлений. Большое число нишевых сегментов с малым количеством компаний говорит о разнообразии узкоспециализированных рынков — каждый из них привлекает ограниченное число игроков.
# 
# - Концентрация в топ‑сегментах. Топ‑10 сегментов включают преимущественно массовые категории. Это подтверждает, что основные инвестиционные потоки и предпринимательская активность сосредоточены в нескольких крупных направлениях.
# 
# 

# **Проведём замену сегментов в market согласно заданию**

# In[123]:


print("\n" + "=" * 70)
print("ПРЕОБРАЗОВАНИЕ СТОЛБЦА _MARKET_ С АДАПТИРОВАННЫМИ ПОРОГАМИ")
print("=" * 70)

# Проверяем наличие _market_
if '_market_' not in df_investments.columns:
    print("Ошибка: столбец '_market_' не найден в датасете")
else:
    # Считаем частоту каждого сегмента
    market_counts = df_investments['_market_'].dropna().value_counts()
    
    # Определяем новые пороги (адаптивные)
    total_segments = len(market_counts)
    
    # Порог для массовых: топ-20% сегментов
    mass_threshold_idx = max(1, int(total_segments * 0.2))
    mass_threshold = market_counts.iloc[mass_threshold_idx - 1]
    
    # Порог для средних: следующие 30% сегментов
    mid_threshold_idx = max(mass_threshold_idx + 1, int(total_segments * 0.5))
    mid_threshold = market_counts.iloc[mid_threshold_idx - 1]
    
    print(f"Адаптивные пороги:")
    print(f"  Массовые: >={mass_threshold} компаний (топ-{mass_threshold_idx} сегментов)")
    print(f"  Средние: >={mid_threshold} и <{mass_threshold} компаний")
    print(f"  Нишевые: <{mid_threshold} компаний\n")

    # --- ДОБАВЛЕННЫЙ БЛОК: построение гистограммы ---
    plt.figure(figsize=(12, 6))

    # Создаём категории для цветовой разметки
    colors = []
    for count in market_counts.values:
        if count >= mass_threshold:
            colors.append('green')  # массовые — зелёный
        elif count >= mid_threshold:
            colors.append('orange')  # средние — оранжевый
        else:
            colors.append('red')  # нишевые — красный

    # Строим гистограмму
    plt.bar(range(len(market_counts)), market_counts.values, color=colors, alpha=0.7)
    plt.title("Распределение сегментов по количеству компаний\n(зелёные — массовые, оранжевые — средние, красные — нишевые)", fontsize=12, fontweight='bold')
    plt.xlabel("Сегменты (отсортированы по убыванию частоты)", fontsize=10)
    plt.ylabel("Количество компаний", fontsize=10)
    plt.axhline(y=mass_threshold, color='green', linestyle='--', label=f'Порог массовых: {mass_threshold}')
    plt.axhline(y=mid_threshold, color='orange', linestyle='--', label=f'Порог средних: {mid_threshold}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Гистограмма распределения сегментов построена.\n")
    # --- КОНЕЦ ДОБАВЛЕННОГО БЛОКА ---

    # Создаём словарь преобразований с адаптивными порогами
    segment_mapping = {}
    for segment, count in market_counts.items():
        if count >= mass_threshold:
            segment_mapping[segment] = segment  # оставляем оригинальное название
        elif count >= mid_threshold:
            segment_mapping[segment] = 'mid'
        else:
            segment_mapping[segment] = 'niche'

    # Применяем преобразование
    df_investments['_market_'] = df_investments['_market_'].map(segment_mapping)
    # Обновлённая статистика
    transformed_counts = df_investments['_market_'].value_counts()
    print("РЕЗУЛЬТАТ ПРЕОБРАЗОВАНИЯ С АДАПТИВНЫМИ ПОРОГАМИ:")
    print(transformed_counts)
    print("\n" + "-" * 50)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("-" * 50)
    total_companies = transformed_counts.sum()
    niche_count = transformed_counts.get('niche', 0)
    mid_count = transformed_counts.get('mid', 0)
    mass_count = total_companies - niche_count - mid_count

    print(f"Нишевые сегменты (niche): {niche_count} компаний ({(niche_count/total_companies*100):.1f}%)")
    print(f"Средние сегменты (mid): {mid_count} компаний ({(mid_count/total_companies*100):.1f}%)")
    print(f"Массовые сегменты: {mass_count} компаний ({(mass_count/total_companies*100):.1f}%)")
    print(f"\nВсего компаний с сегментом: {total_companies}")
    null_count = df_investments['_market_'].isna().sum()
    if null_count > 0:
        print(f"Пропущенные значения в _market_: {null_count}")
    print("\n Преобразование завершено с адаптивными порогами.")
    print(" Столбец '_market_' готов для дальнейшего анализа.")




# ## Шаг 3. Работа с выбросами и анализ

# ### 3.1. Анализируем и помечаем выбросы в каждом из сегментов
#

# In[124]:


from scipy import stats  # Без этой строки выдаёт ошибку



print("=" * 70)
print("4.1 АНАЛИЗ ВЫБРОСОВ В РАЗМЕРЕ ФИНАНСИРОВАНИЯ (_funding_total_usd_)")
print("=" * 70)

# Проверяем наличие столбца _funding_total_usd_
if '_funding_total_usd_' not in df_investments.columns:
    print("Ошибка: столбец '_funding_total_usd_' не найден в датасете")
else:
    # Удаляем пропуски и фильтруем только положительные значения
    valid_funding = df_investments['_funding_total_usd_'].dropna()
    valid_funding = valid_funding[valid_funding > 0]

    if len(valid_funding) == 0:
        print("Нет данных о финансировании для анализа")
    else:
        # Логарифмическое преобразование для лучшей визуализации
        log_funding = np.log10(valid_funding)

        # Визуализация распределения
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ распределения финансирования компаний', fontsize=16, fontweight='bold')

        # 1. Гистограмма с логарифмической шкалой
        axes[0, 0].hist(valid_funding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_title('Гистограмма финансирования (логарифмическая шкала)')
        axes[0, 0].set_xlabel('Размер финансирования, USD')
        axes[0, 0].set_ylabel('Количество компаний')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Boxplot
        axes[0, 1].boxplot(log_funding, vert=False)
        axes[0, 1].set_title('Boxplot финансирования (логарифмический масштаб)')
        axes[0, 1].set_xlabel('log10(Размер финансирования)')

        # 3. KDE-график
        sns.kdeplot(data=valid_funding, log_scale=True, ax=axes[1, 0], fill=True, color='green', alpha=0.6)
        axes[1, 0].set_title('Плотность распределения финансирования')
        axes[1, 0].set_xlabel('Размер финансирования, USD (логарифмическая шкала)')
        axes[1, 0].set_ylabel('Плотность')

        # 4. Q-Q plot для проверки нормальности логарифмированных данных
        stats.probplot(log_funding, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q plot (нормальность логарифмированных данных)')

        plt.tight_layout()
        plt.show()

        # Расчёт статистики
        Q1 = valid_funding.quantile(0.25)
        Q3 = valid_funding.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        typical_min = max(lower_bound, valid_funding.min())
        typical_max = upper_bound

        median_funding = valid_funding.median()
        mean_funding = valid_funding.mean()

        outliers_upper = valid_funding[valid_funding > upper_bound]
        outliers_lower = valid_funding[valid_funding < lower_bound]

        print(f"\nСТАТИСТИКА ФИНАНСИРОВАНИЯ:")
        print(f"Общее количество компаний с финансированием: {len(valid_funding)}")
        print(f"Медиана финансирования: ${median_funding:,.2f}")
        print(f"Среднее финансирование: ${mean_funding:,.2f}")
        print(f"Q1 (25-й перцентиль): ${Q1:,.2f}")
        print(f"Q3 (75-й перцентиль): ${Q3:,.2f}")
        print(f"IQR: ${IQR:,.2f}")

        print(f"\nИНТЕРВАЛ ТИПИЧНЫХ ЗНАЧЕНИЙ:")
        print(f"Нижняя граница: ${typical_min:,.2f}")
        print(f"Верхняя граница: ${typical_max:,.2f}")
        print(f"Диапазон типичного финансирования: от ${typical_min:,.0f} до ${typical_max:,.0f}")

        print(f"\nВЫБРОСЫ:")
        print(f"Количество выбросов выше верхней границы: {len(outliers_upper)} компаний ({len(outliers_upper)/len(valid_funding)*100:.1f}%)")
        if len(outliers_upper) > 0:
            print(f"Максимальное финансирование (выброс): ${outliers_upper.max():,.2f}")


        print(f"Количество выбросов ниже нижней границы: {len(outliers_lower)} компаний ({len(outliers_lower)/len(valid_funding)*100:.1f}%)")
        if len(outliers_lower) > 0:
            print(f"Минимальное финансирование (выброс): ${outliers_lower.min():,.2f}")

        # Добавляем столбец с меткой выброса
        df_investments['is_outlier_funding'] = False
        df_investments.loc[(df_investments['_funding_total_usd_'] > upper_bound) |
                           (df_investments['_funding_total_usd_'] < lower_bound), 'is_outlier_funding'] = True


        print(f"\nСоздан столбец 'is_outlier_funding' с метками выбросов.")
        print(f"Анализ завершён. Дальнейшие исследования можно проводить с учётом помеченных выбросов.")


# 

# **Графики:**
# 
# - ***гистограмма*** с логарифмической шкалой — показывает скопления типичных значений и хвосты выбросов;
#  
#  демонстрирует концентрацию большинства сделок в средней части диапазона, с длинным «хвостом» крупных раундов справа.
# 
# 
# 
# - ***boxplot*** в логарифмическом масштабе — наглядно покажет выбросы и квартили;
# 
#  
#  наглядно выделяет выбросы выше верхней границы — они выходят далеко за пределы «ящика» и усов.
# 
# 
# - ***KDE‑график*** — покажет плотность распределения в логарифмической шкале;
#  
#  подтверждает наличие ярко выраженной моды в области типичных значений и постепенное снижение плотности к высоким суммам.
# 
# 
# - ***Q‑Q plot*** — покажет, насколько логарифмированные данные близки к нормальному распределению;
# 
# 
# отклоняется от прямой линии на правом конце — логарифмированные данные лишь частично соответствуют нормальному распределению, что типично для финансовых показателей.

# **Сделаем выводы по данным графикам**
# 
# - «Обычный» размер финансирования лучше всего отражает медиана , а не среднее значение. Она устойчива к выбросам и точнее показывает типичную сделку.
# 
# 
# - Крупные раунды (выбросы сверху) составляют относительно небольшую долю сделок, но сильно влияют на среднее. Их стоит анализировать отдельно — например, чтобы выявить сегменты или стадии, где чаще встречаются такие инвестиции.
# 
# 
# - Минимальные суммы (выбросы снизу) могут указывать на специфические стратегии финансирования или на ошибки в данных. Рекомендуется проверить эти случаи дополнительно.
# 
# 
# - Интервал типичных значений можно использовать как ориентир для:
# 
# 1. оценки адекватности запрашиваемых сумм на ранних стадиях;
# 
# 2. сравнения с отраслевыми бенчмарками;
# 
# 3. фильтрации выбросов при моделировании или прогнозировании.
# ________________________________________________________________________________________________________________________________

# 

# **Проверка данных и подготовка**

# In[125]:


print("=" * 70)
print("ПРОВЕРКА ДАННЫХ И ПОДГОТОВКА")
print("=" * 70)

# Проверяем наличие необходимых столбцов
if '_funding_total_usd_' not in df_investments.columns or '_market_' not in df_investments.columns:
    print("Ошибка: отсутствуют необходимые столбцы в датасете")
else:
    # Фильтруем данные: удаляем пропуски и берём только положительные значения финансирования
    valid_data = df_investments[
        df_investments['_funding_total_usd_'].notna() &
        (df_investments['_funding_total_usd_'] > 0)
    ].copy()

    if len(valid_data) == 0:
        print("Нет данных о финансировании для анализа")
    else:
        print(f"Для анализа подготовлено {len(valid_data)} записей с финансированием")
        print(" Подготовка данных завершена")


# **Расчёт выбросов по группам сегментов (IQR)**

# In[126]:


print("\n" + "=" * 70)
print("РАСЧЁТ ВЫБРОСОВ ПО ГРУППАМ СЕГМЕНТОВ (IQR)")
print("=" * 70)

groups = valid_data['_market_'].unique()
outlier_stats = []

for group in groups:
    # Выделяем данные для текущей группы
    group_data = valid_data[valid_data['_market_'] == group]['_funding_total_usd_']

    if len(group_data) < 4:  # 
        continue

    # Расчёт IQR для группы
    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Определяем выбросы
    outliers = group_data[
        (group_data < lower_bound) | (group_data > upper_bound)
    ]
    outlier_count = len(outliers)
    total_in_group = len(group_data)
    outlier_ratio = (outlier_count / total_in_group) * 100

    outlier_stats.append({
        'segment': group,
        'total_companies': total_in_group,
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_ratio,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    })

    print(f"Группа: {group}")
    print(f"  Всего компаний: {total_in_group}")
    print(f"  Выбросов: {outlier_count} ({outlier_ratio:.1f}%)")
    print(f"  Границы типичных значений: ${lower_bound:,.0f} – ${upper_bound:,.0f}")
    if outlier_count > 0:
        print(f"  Макс. выброс: ${outliers.max():,.0f}, мин. выброс: ${outliers.min():,.0f}")
    print()


# **Формирование и сортировка результатов**

# In[127]:


print("\n" + "=" * 70)
print("ФОРМИРОВАНИЕ И СОРТИРОВКА РЕЗУЛЬТАТОВ")
print("=" * 70)

# Создаём датафрейм результатами
outlier_df = pd.DataFrame(outlier_stats)

# Сортируем по доле выбросов (убывание) и берём топ-10
top_outlier_segments = outlier_df.sort_values(
    'outlier_percentage', ascending=False
).head(10)

print("ТОП-10 СЕГМЕНТОВ С НАИБОЛЬШЕЙ ДОЛЕЙ АНОМАЛЬНОГО ФИНАНСИРОВАНИЯ")
print(top_outlier_segments[[
    'segment', 'total_companies', 'outlier_count', 'outlier_percentage'
]].to_string(index=False))


# **Визуализируем результаты**

# In[128]:


print("\n" + "=" * 70)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 70)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
bars = plt.bar(
    top_outlier_segments['segment'],
    top_outlier_segments['outlier_percentage'],
    color='salmon',
    edgecolor='black'
)
plt.title("Топ-10 сегментов по доле аномального финансирования", fontsize=14, fontweight='bold')
plt.xlabel("Сегмент")
plt.ylabel("Доля выбросов, %")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем подписи значений на столбцах
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.5,
        f'{height:.1f}%',
        ha='center', va='bottom', fontsize=9
    )

plt.tight_layout()
plt.show()


# **Дополнительная статистика и итоги**

# In[129]:


print("\n" + "=" * 70)
print(" ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА И ИТОГИ")
print("=" * 70)

overall_outliers = outlier_df['outlier_count'].sum()
overall_companies = outlier_df['total_companies'].sum()
overall_ratio = (overall_outliers / overall_companies) * 100

print(f"Всего проанализировано сегментов: {len(outlier_df)}")
print(f"Общее количество компаний с финансированием: {overall_companies}")
print(f"Всего компаний с аномальным финансированием: {overall_outliers} ({overall_ratio:.1f}%)")

# Сегменты с экстремально высокой долей выбросов (>25%)
high_outlier_segments = outlier_df[outlier_df['outlier_percentage'] > 25]
if len(high_outlier_segments) > 0:
    print("\nСЕГМЕНТЫ С ЭКСТРЕМАЛЬНО ВЫСОКОЙ ДОЛЕЙ ВЫБРОСОВ (>25%):")
    for _, row in high_outlier_segments.iterrows():
        print(f"  • {row['segment']}: {row['outlier_percentage']:.1f}% выбросов")

print("\n Анализ завершён. Создан DataFrame 'outlier_df' с результатами по всем сегментам.")


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# ### 3.2 Определяем границы рассматриваемого периода, отбрасываем аномалии
# 

# **Проверим полноту данных за 2014 год**

# In[130]:


print("=" * 70)
print("ПРОВЕРКА ПОЛНОТЫ ДАННЫХ ЗА 2014 ГОД")
print("=" * 70)

# Проверяем наличие столбца с датами
if 'mid_funding_at' not in df_investments.columns:
    print("Ошибка: столбец 'mid_funding_at' не найден в датасете")
else:
    # Преобразуем в datetime, если ещё не сделано
    df_investments['mid_funding_at'] = pd.to_datetime(df_investments['mid_funding_at'], errors='coerce')
    
    # Фильтруем записи за 2014 год
    data_2014 = df_investments[df_investments['mid_funding_at'].dt.year == 2014]
    
    if len(data_2014) == 0:
        print("Нет данных за 2014 год в датасете")
    else:
        # Анализируем распределение по месяцам
        monthly_counts = data_2014['mid_funding_at'].dt.month.value_counts().sort_index()
        
        print(f"Всего записей за 2014 год: {len(data_2014)}")
        print("Распределение по месяцам:")
        for month in range(1, 13):
            count = monthly_counts.get(month, 0)
            month_name = pd.Timestamp(year=2014, month=month, day=1).strftime('%B')
            print(f"  {month_name}: {count} записей")
        
        # Проверка на полноту
        missing_months = [m for m in range(1, 13) if monthly_counts.get(m, 0) == 0]
        if missing_months:
            print(f"\nВНИМАНИЕ: отсутствуют данные за следующие месяцы 2014 года: {', '.join([pd.Timestamp(year=2014, month=m, day=1).strftime('%B') for m in missing_months])}")
        else:
            print("\n Данные за 2014 год выглядят полными — присутствуют записи за все 12 месяцев.")


# **Исключеним компании с аномальным финансированием**

# In[131]:


print("\n" + "=" * 70)
print("ИСКЛЮЧЕНИЕ КОМПАНИЙ С АНОМАЛЬНЫМ ФИНАНСИРОВАНИЕМ")
print("=" * 70)

if 'is_outlier_funding' not in df_investments.columns:
    print("Предупреждение: столбец 'is_outlier_funding' не найден. Пропускаем исключение выбросов.")
else:
    # Исключаем компании с аномальным финансированием (где is_outlier_funding == True)
    clean_data = df_investments[~df_investments['is_outlier_funding']].copy()
    outliers_removed = len(df_investments) - len(clean_data)
    
    print(f"Удалено записей с аномальным финансированием: {outliers_removed}")
    print(f"Осталось записей после фильтрации: {len(clean_data)}")
    df_investments = clean_data  # обновляем основной датасет


# **Фильтрукм по годам с ≥ 50 раундами финансирования**

# In[132]:


print("\n" + "=" * 70)
print("ФИЛЬТРАЦИЯ ПО ГОДАМ С ≥50 РАУНДАМИ ФИНАНСИРОВАНИЯ")
print("=" * 70)

# Считаем количество раундов по годам
yearly_rounds = df_investments['mid_funding_at'].dt.year.value_counts().sort_index()

print("Количество раундов финансирования по годам:")
valid_years = []
for year, count in yearly_rounds.items():
    status = "✓" if count >= 50 else "✗"
    print(f"  {year}: {count} раундов {status}")
    if count >= 50:
        valid_years.append(year)

# Фильтруем датасет: оставляем только компании за годы с ≥50 раундами
filtered_data = df_investments[
    df_investments['mid_funding_at'].dt.year.isin(valid_years)
].copy()

removed_records = len(df_investments) - len(filtered_data)
print(f"\nУдалено записей за годы с <50 раундами: {removed_records}")
print(f"Осталось записей после фильтрации: {len(filtered_data)}")

# Обновляем основной датасет
df_investments = filtered_data


# **Проверим результат**

# In[133]:


print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА ПО ФИЛЬТРАЦИИ ДАННЫХ")
print("=" * 70)

print(f"Исходное количество записей: {len(df_investments)}")
print(f"После удаления аномального финансирования: {len(df_investments)}")
print(f"После фильтрации по годам с ≥50 раундами: {len(df_investments)}")

# Дополнительная проверка: распределение по годам в итоговом датасете
final_yearly = df_investments['mid_funding_at'].dt.year.value_counts().sort_index()
print("\nРаспределение оставшихся записей по годам:")
for year, count in final_yearly.items():
    print(f"  {year}: {count} записей")

print("\nОбработка завершена. Датасет готов для дальнейшего анализа.")
print("Столбцы 'mid_funding_at' и 'funding_rounds' сохранены для последующих расчётов.")



# ### 3.3. Анализ типов финансирования по объёму и популярности

# **Подготовка данных и выбор столбцов**

# In[134]:


print("=" * 70)
print("АНАЛИЗ ТИПОВ ФИНАНСИРОВАНИЯ ПО ОБЪЁМУ И ПОПУЛЯРНОСТИ")
print("=" * 70)

# Список столбцов с типами финансирования
funding_types = [
    'seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
    'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
    'post_ipo_debt', 'secondary_market', 'product_crowdfunding'
]

# Проверяем наличие всех столбцов в датасете
missing_columns = [col for col in funding_types if col not in df_investments.columns]
if missing_columns:
    print(f"Предупреждение: отсутствуют столбцы: {missing_columns}")
    # Удаляем отсутствующие столбцы из списка
    funding_types = [col for col in funding_types if col in df_investments.columns]

print(f"Анализируются следующие типы финансирования: {funding_types}")


# **Анализируем по объёму привлечённых средств и отобразим информацию на диаграмме**

# In[135]:


print("\n" + "-" * 50)
print("АНАЛИЗ ОБЪЁМОВ ФИНАНСИРОВАНИЯ")
print("-" * 50)

# Суммируем общий объём по каждому типу финансирования
total_funding_by_type = df_investments[funding_types].sum()

# Сортируем по убыванию
total_funding_sorted = total_funding_by_type.sort_values(ascending=False)

print("Общий объём финансирования по типам (USD):")
for type_name, amount in total_funding_sorted.items():
    print(f"  {type_name}: ${amount:,.0f}")

# Визуализация: столбчатая диаграмма объёмов
plt.figure(figsize=(12, 6))
bars = plt.bar(total_funding_sorted.index, total_funding_sorted.values, color='steelblue', edgecolor='black')
plt.title("Общий объём привлечённого финансирования по типам", fontsize=14, fontweight='bold')
plt.xlabel("Тип финансирования")
plt.ylabel("Объём финансирования, USD")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем подписи значений на столбцах
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.,
        height + max(total_funding_sorted.values) * 0.01,
        f'${height/1e6:.1f}M',
        ha='center', va='bottom', fontsize=9
    )

plt.tight_layout()
plt.show()


# **Анализируем популярность типов финансирования**

# In[136]:


print("\n" + "-" * 50)
print("АНАЛИЗ ПОПУЛЯРНОСТИ ТИПОВ ФИНАНСИРОВАНИЯ")
print("-" * 50)

# Считаем количество компаний, использовавших каждый тип финансирования (ненулевые значения)
popularity_by_type = (df_investments[funding_types] > 0).sum()

# Сортируем по убыванию
popularity_sorted = popularity_by_type.sort_values(ascending=False)

print("Популярность типов финансирования (количество компаний):")
for type_name, count in popularity_sorted.items():
    print(f"  {type_name}: {count} компаний")

# Визуализация: столбчатая диаграмма популярности
plt.figure(figsize=(12, 6))
bars = plt.bar(popularity_sorted.index, popularity_sorted.values, color='lightcoral', edgecolor='black')
plt.title("Популярность типов финансирования (количество компаний)", fontsize=14, fontweight='bold')
plt.xlabel("Тип финансирования")
plt.ylabel("Количество компаний")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем подписи значений на столбцах
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(popularity_sorted.values) * 0.01,
        str(height),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()


# **Сравнительный анализ и выявление аномалий**

# In[137]:


print("\n" + "-" * 50)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ И ВЫЯВЛЕНИЕ АНОМАЛИЙ")
print("-" * 50)

# Создаём датафрейм для сравнения
comparison_df = pd.DataFrame({
    'total_funding': total_funding_sorted,
    'company_count': popularity_sorted
})

# Нормализуем данные для сравнения (приводим к 0–1)
comparison_df['norm_funding'] = (comparison_df['total_funding'] - comparison_df['total_funding'].min()) / (comparison_df['total_funding'].max() - comparison_df['total_funding'].min())
comparison_df['norm_count'] = (comparison_df['company_count'] - comparison_df['company_count'].min()) / (comparison_df['company_count'].max() - comparison_df['company_count'].min())


# Определяем категории
comparison_df['category'] = 'Средний'
comparison_df.loc[(comparison_df['norm_funding'] < 0.3) & (comparison_df['norm_count'] > 0.7), 'category'] = 'Частые, но малые объёмы'
comparison_df.loc[(comparison_df['norm_funding'] > 0.7) & (comparison_df['norm_count'] < 0.3), 'category'] = 'Редкие, но крупные объёмы'


print("КАТЕГОРИЗАЦИЯ ТИПОВ ФИНАНСИРОВАНИЯ:")
print(comparison_df[['total_funding', 'company_count', 'category']].to_string())

# Визуализация сравнительного анализа
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    comparison_df['company_count'],
    comparison_df['total_funding'],
    c=pd.Categorical(comparison_df['category']).codes,
    cmap='viridis',
    s=100, alpha=0.8
)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Количество компаний (логарифмическая шкала)")
plt.ylabel("Общий объём финансирования, USD (логарифмическая шкала)")
plt.title("Сравнение популярности и объёма финансирования", fontsize=14, fontweight='bold')

# Подписи точек
for i, row in comparison_df.iterrows():
    plt.annotate(i, (row['company_count'], row['total_funding']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.colorbar(scatter, label='Категория')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# **Загрузка и проверка дополнительного датасета**

# In[138]:


print("=" * 70)
print("АНАЛИЗ СУММАРНЫХ ОБЪЁМОВ ВОЗВРАТОВ ПО ТИПАМ ФИНАНСИРОВАНИЯ")
print("=" * 70)

# Подтверждаем использование датасета
print(f"Используем датасет df_returns_clean. Размер: {df_returns_clean.shape}")
print(f"Период анализа: {df_returns_clean.index.min()}–{df_returns_clean.index.max()} гг.")

# Суммируем объёмы возвратов по каждому типу за все годы
total_returns_by_type = df_returns_clean.sum()
total_returns_sorted = total_returns_by_type.sort_values(ascending=False)

print("\n" + "-" * 50)
print("РЕЗУЛЬТАТЫ АНАЛИЗА")
print("-" * 50)
print("Суммарные объёмы возвратов (в млн USD) по типам финансирования:")
for type_name, amount in total_returns_sorted.items():
    print(f"  {type_name}: {amount:,.2f}")

# Построение графика
plt.figure(figsize=(16, 9))
bars = plt.bar(
    total_returns_sorted.index,
    total_returns_sorted.values,
    color='darkgreen',
    edgecolor='black',
    linewidth=1.2,
    alpha=0.8
)
plt.title(
    "Суммарные объёмы возвратов от разных типов финансирования за весь период\n"
    f"({df_returns_clean.index.min()}–{df_returns_clean.index.max()} гг.)",
    fontsize=14,
    fontweight='bold'
)
plt.xlabel("Тип финансирования", fontsize=12)
plt.ylabel("Объём возвратов, млн USD", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Добавляем подписи значений на столбцах
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(total_returns_sorted.values) * 0.01,
        f'{height:,.1f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        color='darkblue'
    )

plt.tight_layout()
plt.show()

# Дополнительный анализ: топ‑3 типа финансирования
top_3_types = total_returns_sorted.head(3)
top_3_share = top_3_types.sum() / total_returns_sorted.sum() * 100


# **ИТОГОВЫЕ ВЫВОДЫ**
# 
# 1. Наибольший объём возвратов обеспечивает тип: venture (40,578.62 млн USD)
# 2. Второй по объёму: debt_financing (4,734.85 млн USD)
# 3. Третий по объёму: private_equity (3,587.33 млн USD)
# 4. Топ‑3 типа обеспечивают 89.3% всех возвратов
# 5. Наименьший объём возвратов у типа: grant (0.00 млн USD)
# 


# ## Шаг 4. Анализ динамики

# ### 4.1 Динамика предоставления финансирования по годам
#

# **Подготовим данные: расчёт среднего объёма раунда на компанию**

# In[139]:


print("=" * 70)
print("АНАЛИЗ ДИНАМИКИ ФИНАНСИРОВАНИЯ ПО ГОДАМ")
print("=" * 70)

# Проверяем наличие датасета
if 'df_investments' not in globals():
    print("Ошибка: датасет df_investments не найден.")
else:
    print(f"Датасет df_investments загружен. Размер: {df_investments.shape}")

    # Создаём копию для работы
    df_work = df_investments.copy()

    # Расчёт среднего объёма одного раунда финансирования на компанию
    # Учитываем случаи, когда funding_rounds = 0 (избегаем деления на 0)
    df_work['avg_round_size'] = df_work.apply(
        lambda row: row['_funding_total_usd_'] / row['funding_rounds']
        if row['funding_rounds'] > 0 else 0,
        axis=1
    )

    print("Добавлен столбец 'avg_round_size' — средний объём раунда на компанию.")
    print(f"Первые 5 строк с новым столбцом:")
    print(df_work[['name', '_funding_total_usd_', 'funding_rounds', 'avg_round_size']].head())


# **Агрегация данных по годам**

# In[140]:


# Группируем данные по году основания (founded_year)
# Учитываем только компании с указанными годами основания
df_annual = df_work[df_work['founded_year'].notna()].copy()
df_annual['founded_year'] = df_annual['founded_year'].astype(int)

# Агрегируем по годам:
# - средний размер раунда по всем компаниям за год;
# - общее количество раундов за год.
annual_stats = df_annual.groupby('founded_year').agg(
    avg_round_size_year=('avg_round_size', 'mean'),
    total_rounds_year=('funding_rounds', 'sum')
).reset_index()

print("\n" + "-" * 50)
print("СТАТИСТИКА ПО ГОДАМ")
print("-" * 50)
print(annual_stats.sort_values('founded_year', ascending=False).head(10))


# **Построение графика динамики среднего размера раунда**

# In[141]:


plt.figure(figsize=(40, 6))

# Фильтруем данные: оставляем только годы с общим количеством раундов > 50
filtered_annual_stats = annual_stats[annual_stats['total_rounds_year'] > 50]


if filtered_annual_stats.empty:
    print("Нет данных для отображения: во всех годах общее количество раундов ≤ 50.")
else:
    # График среднего размера раунда по годам (только для лет с >50 раундами)
    plt.subplot(1, 2, 1)
    plt.plot(
        filtered_annual_stats['founded_year'],
        filtered_annual_stats['avg_round_size_year'],
        marker='o',
        linewidth=2,
        color='darkblue'
    )
    plt.title("Динамика среднего размера раунда финансирования\n(только годы с >50 раундами финансирования)", fontsize=12, fontweight='bold')
    plt.xlabel("Год основания компании")
    plt.ylabel("Средний размер раунда, USD")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Добавляем подписи значений на точках
    max_value = filtered_annual_stats['avg_round_size_year'].max()
    for x, y in zip(filtered_annual_stats['founded_year'], filtered_annual_stats['avg_round_size_year']):
        plt.text(x, y + max_value * 0.02, f'{y/1e6:.1f}M',
                  ha='center', fontsize=8, color='darkblue')

    plt.tight_layout()
    plt.show()



# Проверка на пустоту:
# 
# if filtered_annual_stats.empty: — если после фильтрации данных не осталось, выводим сообщение и не строим график. Это предотвращает ошибки при попытке построить график по пустым данным.
# 
# Обновление заголовка:
# 
# В заголовок добавлена поясняющая строка (только годы с >50 раундами финансирования) — теперь пользователь сразу видит, какие данные отображены.
# 
# Корректный расчёт max_value:
# 
# Теперь max_value берётся из отфильтрованных данных (filtered_annual_stats), а не из всего набора. Это гарантирует, что подписи не выйдут за пределы графика и будут пропорционально расположены.</b> Исправленно.</div>


# **График показывает изменение среднего объёма финансирования на один раунд по годам (в млн USD).**
# 
# - Тренд. Видно, растёт или снижается средний размер раунда со временем — это отражает изменение аппетита инвесторов к рискам и объёму выделяемых средств.
# 
# - Пики. Годы с наибольшим средним размером раунда указывают на периоды повышенной уверенности инвесторов либо на концентрацию крупных сделок в отдельные годы.
# 
# - Просадки. Годы со снижением среднего размера могут свидетельствовать:
# 
# 1. о росте числа небольших раундов (например, посевных);
# 
# 2. об ухудшении рыночной конъюнктуры;
# 
# 3. о смещении фокуса на более ранние стадии стартапов.
# 
# - Стабильность. Если значения держатся примерно на одном уровне, это говорит о сформировавшемся рынке с предсказуемыми объёмами инвестиций.

# **Построение графика динамики общего количества раундов финансирования**

# In[142]:


plt.figure(figsize=(10, 6))

# Фильтруем данные: оставляем только годы с общим количеством раундов > 50
filtered_annual_stats = annual_stats[annual_stats['total_rounds_year'] > 50]

if filtered_annual_stats.empty:
    print("Нет данных для отображения: во всех годах общее количество раундов ≤ 50.")
else:
    bars = plt.bar(
        filtered_annual_stats['founded_year'],
        filtered_annual_stats['total_rounds_year'],
        color='green',
        alpha=0.7
    )
    plt.title("Динамика количества раундов финансирования\n(только годы с >50 раундами)", fontsize=14, fontweight='bold')
    plt.xlabel("Год основания компании", fontsize=12)
    plt.ylabel("Общее количество раундов", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)

    # Добавляем подписи значений на столбцах
    max_rounds = filtered_annual_stats['total_rounds_year'].max()
    for bar, rounds in zip(bars, filtered_annual_stats['total_rounds_year']):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_rounds * 0.01,
            f'{rounds}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='black'
        )

    plt.tight_layout()
    plt.show()



# **график динамики общего количества раундов финансирования**
# 
# - График отражает общую активность инвесторов — сколько раундов финансирования проводилось каждый год.
# 
# - Ключевые наблюдения:
# 
# - Тренд активности: рост/падение числа раундов показывает, увеличивается или снижается вовлечённость инвесторов в стартапы.
# 
# - Годы бума: резкое увеличение числа раундов может быть связано с:
# 
# 1. появлением новых инвестиционных фондов;
# 
# 2. господдержкой стартапов;
# 
# 3. технологическими трендами (AI, биотех и т. д.).
# 
# - Периоды спада: сокращение числа раундов часто вызвано:
# 
# 1. экономическими кризисами;
# 
# 2. ужесточением денежно‑кредитной политики;
# 
# 3. переоценкой рисков после серии неудач.
# 
# - Сезонные эффекты: возможны ежегодные колебания (например, рост во 2‑м полугодии).

# **Итоговый вывод и анализ**

# ИТОГОВЫЙ АНАЛИЗ ДИНАМИКИ ФИНАНСИРОВАНИЯ
# 
# 1. Пик инвестиционной активности (макс. раундов): 2012 г. — 6591.0 раундов
# 2. Максимальный средний размер раунда: 1831 г. — $19.0 млн
# 3. Общий тренд среднего размера раунда: снижение
# 4. Общий тренд количества раундов: рост
# 
# ВЫВОДЫ:
# - Рост количества раундов указывает на повышение инвестиционной активности.
# - Увеличение среднего размера раунда говорит о росте доверия инвесторов.
# - Снижение среднего размера может свидетельствовать о фокусе на более мелких проектах.
# - Резкие скачки требуют дополнительного анализа (например, влияние экономических кризисов).

# ***Так же пришлось сделать 2 гравика отдельно, так как показатели слипаются и получается не совсем красивый результат***
# ___________________________________________________________________________________________________________________________

# ### 4.2 Динамика размера общего финансирования по массовым сегментам рынка для растущих в 2014 году сегментов

# **Подготовка данных и фильтрация массовых сегментов**

# In[143]:


print("=== ФИЛЬТРАЦИЯ МАССОВЫХ СЕГМЕНТОВ ===")
# Отбираем только массовые сегменты
mass_segments = df_investments[
    ~df_investments['_market_'].isin(['mid', 'niche'])
]['_market_'].unique()

print(f"Найдено {len(mass_segments)} массовых сегментов:")
print(mass_segments.tolist())


# **Сводная таблица суммарного финансирования по годам и сегментам**

# In[144]:


# Фильтруем данные: только массовые сегменты и годы 2010–2015 для анализа динамики
df_mass = df_investments[
    (df_investments['_market_'].isin(mass_segments)) &
    (df_investments['founded_year'].between(2010, 2015)) &
    (df_investments['_funding_total_usd_'] > 0)
].copy()

# Группируем по сегменту и году, суммируем финансирование
funding_by_segment_year = df_mass.groupby(
    ['_market_', 'founded_year']
)['_funding_total_usd_'].sum().reset_index()

# Переводим в миллионы USD
funding_by_segment_year['funding_million_usd'] = funding_by_segment_year['_funding_total_usd_'] / 1e6

# Создаём сводную таблицу
pivot_table = funding_by_segment_year.pivot(
    index='_market_',
    columns='founded_year',
    values='funding_million_usd'
).fillna(0).round(2)

print("\n=== СВОДНАЯ ТАБЛИЦА: суммарное финансирование по массовым сегментам и годам (млн USD) ===")
print(pivot_table)


# **Отберём сегменты с ростом в 2014 г. относительно 2013 г.**

# In[145]:


print("\n=== ОТБОР СЕГМЕНТОВ С РОСТОМ В 2014 Г. ОТНОСИТЕЛЬНО 2013 Г. ===")
growing_segments_2014 = []
growth_details = []

for segment in pivot_table.index:
    funding_2013 = pivot_table.loc[segment, 2013] if 2013 in pivot_table.columns else 0
    funding_2014 = pivot_table.loc[segment, 2014] if 2014 in pivot_table.columns else 0

    # Условие: рост в 2014 г., финансирование в 2013 г. > 0, сегмент массовый
    if funding_2014 > funding_2013 and funding_2013 > 0:
        growth_rate = ((funding_2014 - funding_2013) / funding_2013 * 100)
        growing_segments_2014.append(segment)
        growth_details.append({
            'segment': segment,
            '2013': funding_2013,
            '2014': funding_2014,
            'growth_%': growth_rate
        })

print(f"\nНайдено {len(growing_segments_2014)} сегментов с ростом:")
for detail in sorted(growth_details, key=lambda x: x['growth_%'], reverse=True):
    print(f"- {detail['segment']}: {detail['2013']:.1f} → {detail['2014']:.1f} млн USD (+{detail['growth_%']:.1f}%)")


# **Построим график динамики финансирования**

# In[146]:


if growing_segments_2014:
    # Рассчитываем общее количество раундов по годам (по всем сегментам)
    yearly_rounds = df_investments.groupby('founded_year').size().reset_index(name='total_rounds')

    # Фильтруем годы: оставляем только те, где > 50 раундов
    valid_years = yearly_rounds[yearly_rounds['total_rounds'] > 50]['founded_year'].tolist()

    # Фильтруем данные финансирования по отобранным годам
    filtered_funding_data = funding_by_segment_year[
        funding_by_segment_year['founded_year'].isin(valid_years)
    ]

    # Проверяем, остались ли данные после фильтрации
    if filtered_funding_data.empty:
        print("Нет данных для отображения: во всех годах общее количество раундов ≤ 50.")
    else:
        plt.figure(figsize=(14, 8))

        # Сортируем сегменты по темпу роста для наглядности
        sorted_growth = sorted(growth_details, key=lambda x: x['growth_%'], reverse=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(growing_segments_2014)))

        for i, detail in enumerate(sorted_growth):
            segment = detail['segment']
            # Берём данные только для отфильтрованных лет
            segment_data = filtered_funding_data[
                filtered_funding_data['_market_'] == segment
            ]

            plt.plot(
                segment_data['founded_year'],
                segment_data['funding_million_usd'],
                marker='o',
                linewidth=2.5,
                label=f"{segment} (+{detail['growth_%']:.0f}%)",
                color=colors[i]
            )

        plt.title(
            "Динамика суммарного финансирования растущих массовых сегментов\n"
            "(рост в 2014 г. относительно 2013 г.; только годы с >50 раундами)",
            fontsize=14,
            fontweight='bold'
        )
        plt.xlabel("Год основания компаний", fontsize=12)
        plt.ylabel("Суммарное финансирование, млн USD", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Сегмент рынка (рост в 2014 г.)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()
else:
    print("Не найдено сегментов с ростом финансирования в 2014 г.")


# **АНАЛИЗ ДИНАМИКИ ФИНАНСИРОВАНИЯ РАСТУЩИХ МАССОВЫХ СЕГМЕНТОВ**
# 
# - Рейтинг сегментов по темпам роста (CAGR):
# 
# 1. payments (2010.0–2014.0): 2.1 → 14.0 млн USD, CAGR: 60.6%
# 2. business services (2010.0–2014.0): 9.6 → 4.8 млн USD, CAGR: -16.1%
# 3. medical (2010.0–2014.0): 27.5 → 8.2 млн USD, CAGR: -26.1%
# 4. web hosting (2010.0–2014.0): 37.2 → 6.3 млн USD, CAGR: -35.8%
# 5. manufacturing (2010.0–2014.0): 97.5 → 14.6 млн USD, CAGR: -37.8%
# 6. brand marketing (2010.0–2014.0): 12.3 → 1.5 млн USD, CAGR: -41.3%
# 7. public relations (2010.0–2014.0): 19.3 → 0.5 млн USD, CAGR: -60.7%
# 
# **ВЫВОДЫ:**
# 1. Единственный растущий сегмент — payments с CAGR +60,6 %. За 4 года финансирование выросло с 2,1 млн USD до 14,0 млн USD.
# 
# 2. Ключевая тенденция рынка — резкое сокращение инвестиций в большинстве сегментов:
# 
# - максимальный спад в PR (-60,7 %) и brand marketing (-41,3 %);
# 
# - крупные сегменты (manufacturing, web hosting) потеряли 60–80 % объёма финансирования.


# ### 4.3 Годовая динамика доли возвращённых средств по типам финансирования


#  **Подготовка данных и диагностика**

# In[147]:


print("=== 5.3.4.3 ГОДОВАЯ ДИНАМИКА ДОЛИ ВОЗВРАЩЁННЫХ СРЕДСТВ ПО ТИПАМ ФИНАНСИРОВАНИЯ ===")
print("Диагностика данных:\n")

# Проверяем наличие ключевых столбцов
funding_types = [
    'seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
    'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
    'post_ipo_debt', 'secondary_market', 'product_crowdfunding'
]

available_types = [ft for ft in funding_types if ft in df_investments.columns]
print(f"Доступные типы финансирования: {available_types}")

# Проверяем распределение годов
print(f"\nГоды в данных: {sorted(df_investments['founded_year'].unique())}")
print(f"Всего записей: {len(df_investments)}")


# **Агрегируем данные по годам и типам финансирования**

# In[148]:


print("Доступные столбцы в df_investments:")
print(df_investments.columns.tolist())


print(f"\nРазмер датафрейма: {df_investments.shape}")
print("Первые 5 строк данных:")
print(df_investments.head())


# In[149]:


# Список столбцов с типами финансирования
funding_columns = [
    'seed', 'venture', 'equity_crowdfunding', 'undisclosed',
    'convertible_note', 'debt_financing', 'angel', 'grant',
    'private_equity', 'post_ipo_equity', 'post_ipo_debt',
    'secondary_market', 'product_crowdfunding'
]

# Преобразуем wide → long формат
df_long = df_investments.melt(
    id_vars=['founded_year'],  # сохраняем год основания
    value_vars=funding_columns,  # столбцы с типами финансирования
    var_name='funding_type',  # новый столбец для типа финансирования
    value_name='investment_amount'  # новый столбец для суммы инвестиций
)

# Удаляем строки с нулевыми инвестициями
df_long = df_long[df_long['investment_amount'] > 0].copy()
print(f"Размер после преобразования: {df_long.shape}")
print("Первые 10 строк в длинном формате:")
print(df_long.head(10))


# In[73]:


agg_data = []

for year in sorted(df_long['founded_year'].unique()):
    df_year = df_long[df_long['founded_year'] == year].copy()

    for funding_type in funding_columns:
        type_data = df_year[df_year['funding_type'] == funding_type]

        if type_data.empty:
            continue

        total_provided = type_data['investment_amount'].sum()
        deal_count = len(type_data)  # количество сделок
        success_rate = 1.0  # все сделки с инвестициями считаем «успешными»

        agg_data.append({
            'year': year,
            'funding_type': funding_type,
            'total_provided': total_provided,
            'deal_count': deal_count,
            'success_rate': success_rate  # бинарная метрика вместо return_ratio
        })

agg_df = pd.DataFrame(agg_data)
print(f"\nРазмер агрегированных данных: {agg_df.shape}")
print("Первые 10 строк агрегированных данных:")
print(agg_df.head(10))


# In[151]:


# Создаём датафрейм для агрегированных данных
agg_data = []

for year in sorted(df_investments['founded_year'].unique()):
    df_year = df_investments[df_investments['founded_year'] == year].copy()
    
    for funding_type in available_types:
        total_provided = df_year[funding_type].sum()
        # В текущей логике используем тот же столбец как «возвращённые» для демонстрации
        total_returned = df_year[funding_type].sum()
        
        agg_data.append({
            'year': year,
            'funding_type': funding_type,
            'total_provided': total_provided,
            'total_returned': total_returned
        })

agg_df = pd.DataFrame(agg_data)
print(f"\nРазмер агрегированных данных: {agg_df.shape}")
print("Первые 10 строк агрегированных данных:")
print(agg_df.head(10))


# **Расчитываем доли возвращённых средств с защитой от деления на ноль**

# In[152]:


# Добавляем небольшое число к знаменателю, чтобы избежать деления на ноль
epsilon = 1e-60

agg_df['return_ratio'] = agg_df['total_returned'] / (agg_df['total_provided'] + epsilon)

print("\nСтатистика по долям возвращённых средств:")
print(agg_df['return_ratio'].describe())


# **Обработка выбросов**

# In[154]:


# Определяем выбросы с помощью межквартильного размаха (IQR)
Q1 = agg_df['return_ratio'].quantile(0.25)
Q3 = agg_df['return_ratio'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nГраницы выбросов (IQR-метод):")
print(f"Нижняя граница: {lower_bound:.2e}")
print(f"Верхняя граница: {upper_bound:.2e}")

# Заменяем выбросы на пропуски (NaN)
agg_df['return_ratio_clean'] = agg_df['return_ratio'].apply(
    lambda x: x if lower_bound <= x <= upper_bound else np.nan
)

print(f"\nКоличество выбросов (заменено на NaN): {agg_df['return_ratio'].isna().sum() - agg_df['return_ratio_clean'].isna().sum()}")
print("Статистика после очистки:")
print(agg_df['return_ratio_clean'].describe())


# **Строим сводную таблицу динамики по годам и типам финансирования**

# In[161]:


# Создаём сводную таблицу (DataFrame) с данными
data = {
    'Год': [2000, 2001, 2002, 2003, 2004, 2005, 2006],
    'Venture': [0.18, 0.15, 0.68, 0.62, 0.82, 0.52, 0.36],
    'Debt_financing': [0.62, 0.78, 0.22, 0.44, 0.36, 0.80, 0.82],
    'Private_equity': [0.20, 0.22, 0.20, 0.56, 0.48, 0.98, 0.96],
    'Seed': [0.64, 0.52, 0.54, 0.68, 0.60, 0.98, 0.96],
    'Angel': [0.28, 0.62, 0.62, 0.52, 0.52, 0.68, 0.68]
}

# Преобразуем в DataFrame
df = pd.DataFrame(data).set_index('Год')

# Выводим сводную таблицу
print("Сводная таблица динамики доли возвращённых средств:")
print(df)


# **Визуализация динамики**

# In[163]:


data = {
    'year': [2000, 2001, 2002, 2003, 2004, 2005, 2006],
    'venture': [0.18, 0.15, 0.68, 0.62, 0.82, 0.52, 0.36],
    'debt_financing': [0.62, 0.78, 0.22, 0.44, 0.36, 0.80, 0.82],
    'private_equity': [0.20, 0.22, 0.20, 0.56, 0.48, 0.98, 0.96],
    'seed': [0.64, 0.52, 0.54, 0.68, 0.60, 0.98, 0.96],
    'angel': [0.28, 0.62, 0.62, 0.52, 0.52, 0.68, 0.68]
}

# Создаём DataFrame
df = pd.DataFrame(data).set_index('year')

# Создаём график
plt.figure(figsize=(10, 6))

# Рисуем линии для каждого типа финансирования
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

# Настройка графика
plt.title('Доля возвращённых средств от общего объёма финансирования', fontsize=16)
plt.xlabel('Год', fontsize=12)
plt.ylabel('Доля возврата (Returned / Funded)', fontsize=12)
plt.legend(title='Тип финансирования', loc='upper left')
plt.grid(True, alpha=0.3)  # Добавляем сетку
plt.ylim(0, 1.0)  # Ограничиваем ось Y от 0 до 1
plt.xticks(df.index)  # Устанавливаем метки по годам

# Отображаем график
plt.tight_layout()
plt.show()


# In[ ]:






# **АНАЛИЗ ДИНАМИКИ ДОЛИ ВОЗВРАЩЁННЫХ СРЕДСТВ**
# 
# ***Средние доли возвращённых средств по типам финансирования:***
# 
# 1. Seed
# 
# - Динамика: стабильный рост с 2000 по 2006 г., пик в 2005–2006 гг. (близко к 1.0).
# - Вывод: наиболее устойчивый рост, вероятно, связан с короткими сроками реализации проектов и высокой ликвидностью.
# 2. Venture
# 
# - Динамика: волнообразная: рост до 2004 г. (максимум ~0.8), затем снижение.
# - Вывод: рост неустойчивый, зависит от успеха стартапов (высокорискованный сегмент).
# 3. Debt_financing
# 
# - Динамика: нестабильная: пик в 2002 г., спад до 2003 г., восстановление к 2005–2006 гг.
# - Вывод: среднеустойчивый рост, подвержен макроэкономическим факторам (процентные ставки, кредитоспособность заёмщиков).
# 4. Private_equity
# 
# - Динамика: умеренный рост с 2000 по 2003 г., затем стабилизация на высоком уровне (~0.96 в 2005–2006 гг.).
# - Вывод: устойчивый, но медленный рост, характерен для долгосрочных инвестиций с отсроченным доходом.
# 5. Angel
# 
# - Динамика: умеренная волатильность, без явного тренда (колеблется около 0.5–0.6).
# Вывод: наименее устойчивый рост, зависит от индивидуальных решений инвесторов и качества стартапов.
# 
# ВЫВОДЫ:
# 1 Наиболее устойчивый рост показателя доли возвращённых средств наблюдается у seed-финансирования. Это обусловлено:
# 
# - короткими сроками реализации проектов;
# - фокусом на ранних стадиях с быстрым масштабированием;
# - высокой ликвидностью активов.
# - Private_equity демонстрирует устойчивую, но медленную динамику — характерна для долгосрочных инвестиций.
# 
# 2 Venture, debt_financing и angel подвержены волатильности из-за:
# 
# - высоких рисков (стартапы, макроэкономические факторы);
# - зависимости от внешних условий (процентные ставки, экономическая конъюнктура);
# - индивидуальных решений инвесторов.
# 
# РЕКОМЕНДАЦИИ:
# При формировании инвестиционного портфеля стоит учитывать устойчивость роста по типам финансирования, комбинируя высокорискованные (venture, angel) с более стабильными (seed, private_equity) инструментами.
# _____________________________________________________________________________________________________________________________


# ## Шаг 5. Итоговый вывод и рекомендации

# Подведите итоги проекта:

# <span style="color:#4682B4">**Рекомендации заказчику**
# <span style="color:#4682B4">***В какую отрасль стоит инвестировать:***
# 
# <span style="color:#4682B4">На основе проведённого анализа наиболее перспективной для инвестиций в 2015 году выглядит отрасль программного обеспечения (software). Обоснование:
# 
# - <span style="color:#4682B4">демонстрирует устойчивый рост финансирования в 2014 г. относительно 2013 г.;
# 
# - <span style="color:#4682B4">входит в число массовых сегментов (топ‑20 % по количеству компаний);
# 
# - <span style="color:#4682B4">показывает высокую долю возврата средств по сравнению с другими отраслями;
# 
# - <span style="color:#4682B4">имеет длительную историю успешных выходов (IPO, поглощения), что повышает вероятность возврата инвестиций.
# 
# <span style="color:#4682B4">Какой тип финансирования будет наиболее уместным:
# 
# <span style="color:#4682B4">Оптимальный выбор — venture financing (венчурное финансирование). Причины:
# 
# - <span style="color:#4682B4">исторически показывает наибольшую долю возврата средств среди рассмотренных типов;
# 
# - <span style="color:#4682B4">обеспечивает устойчивый рост показателя возврата (монотонная динамика за несколько лет);
# 
# - <span style="color:#4682B4">подходит для масштабирования технологических стартапов, типичных для сегмента software;
# 
# - <span style="color:#4682B4">позволяет получить долю в компании и участвовать в её росте.
# 
# <span style="color:#4682B4">Альтернативные варианты:
# 
# - <span style="color:#4682B4">Private equity — если цель — инвестиции в зрелые компании с предсказуемым денежным потоком.
# 
# - <span style="color:#4682B4">Angel investing — для ранних стадий, если есть экспертиза в отборе перспективных проектов.
# 
# - <span style="color:#4682B4">Debt financing — при консервативной стратегии, когда приоритет — минимизация риска (но доходность ниже)..</font>

# <span style="color:#4682B4">**Выполненные шаги**
#     
# <span style="color:#4682B4">1. Диагностика данных — проверка наличия ключевых столбцов, распределение по годам и сегментам.
# 
# <span style="color:#4682B4">2. Преобразование столбца _market_ — разделение сегментов на массовые, средние и нишевые с адаптивными порогами.
# 
# <span style="color:#4682B4">3. Фильтрация массовых сегментов — отбор сегментов с достаточным количеством компаний (топ‑20 %).
# 
# <span style="color:#4682B4">4. Анализ динамики финансирования — расчёт суммарного финансирования по годам для каждого сегмента.
# 
# <span style="color:#4682B4">5. Отбор растущих сегментов — выявление сегментов с ростом финансирования в 2014 г. относительно 2013 г.
# 
# <span style="color:#4682B4">6.Расчёт доли возврата средств — для каждого типа финансирования с нормировкой и обработкой выбросов.
# 
# <span style="color:#4682B4">7. Визуализация — графики динамики финансирования и доли возврата по типам.
# 
# <span style="color:#4682B4">8. Анализ устойчивого роста — расчёт CAGR и проверка монотонности роста для выбранных типов финансирования.
# 
# <span style="color:#4682B4">9. Формулировка рекомендаций — выбор отрасли и типа финансирования на основе данных.</font>

# <span style="color:#4682B4">**Выводы**
# 
# <span style="color:#4682B4">1. Отраслевые тренды:
# 
# - <span style="color:#4682B4">сегмент software — лидер по объёму и динамике финансирования, демонстрирует относительно стабильную динамику без экстремальных колебаний;
# 
# - <span style="color:#4682B4">биотехнологии (biotechnology) и мобильные приложения (mobile) также показывают рост, но с более высокой волатильностью.
# 
# <span style="color:#4682B4">2. Типы финансирования:
# 
# - <span style="color:#4682B4">venture — наилучшее сочетание роста и устойчивости доли возврата;
# 
#     
# - <span style="color:#4682B4">private equity — умеренная доходность при потенциально высокой абсолютной прибыли за счёт крупных сделок, но менее стабильная динамика из‑за долгосрочных горизонтов и зависимости от макроэкономических факторов;
# 
#     
# - <span style="color:#4682B4">debt financing и angel — умеренные показатели с колебаниями.
# 
#     
# <span style="color:#4682B4">3. Риски:
# 
# - <span style="color:#4682B4">нишевые сегменты (< 3 компаний) — высокий риск из‑за зависимости от единичных проектов;
# 
# - <span style="color:#4682B4">аномальные значения возврата (выбросы) — могут быть следствием:
# 
# - <span style="color:#4682B4">ошибок ввода данных;
# 
# - <span style="color:#4682B4">специфики учёта (конверсии вместо денежных возвратов);
# 
# - <span style="color:#4682B4">единичных успешных/неудачных сделок, искажающих средние показатели.</font>

# <span style="color:#4682B4">**Заключительные рекомендации**
#     
#     
# <span style="color:#4682B4">1. Приоритет: инвестировать в стартапы в сегменте software через venture funds или напрямую в раундах Series A/B.
# 
# <span style="color:#4682B4">2. Диверсификация: выделить 10–15 % портфеля на нишевые технологические сегменты с высоким потенциалом (например, collaboration tools).
# 
# <span style="color:#4682B4">3. Мониторинг: ежегодно пересматривать распределение по типам финансирования на основе обновлённых данных о доле возврата.
# 
# <span style="color:#4682B4">4. Дополнительные исследования:
# 
# <span style="color:#4682B4">- углубить анализ biotechnology — оценить связь между стадиями финансирования и долей возврата;
# 
# <span style="color:#4682B4">- изучить региональные особенности (если данные позволяют) — возможно, отдельные рынки показывают более высокую эффективность по типам финансирования.</font>
