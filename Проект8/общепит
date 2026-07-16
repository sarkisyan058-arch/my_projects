
# ## 1 Загрузка данных

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import phik
# Загрузка данных
rest_info = pd.read_csv('/datasets/rest_info.csv')
rest_price = pd.read_csv('/datasets/rest_price.csv')


  

# - Знакомимся с данными

# In[3]:


# Выволим информацию о rest_info и первые 5 строк данного датасета
print("Информация о rest_info:")
display(rest_info.info())

print("\nПервые 5 строк rest_info:")
display(rest_info.head())

print(f"\nРазмер датасета rest_info: {rest_info.shape}")

print("\nОбщая информация:")
display(rest_info.info())



# In[4]:


# Выволим информацию о rest_price и первые 5 строк данного датасета
print("Информация о rest_price:")
print(rest_price.info())
print("\nПервые 5 строк rest_price:")
print(rest_price.head())
print("\nРазмер датасета rest_price:", rest_price.shape)
print("\nОбщая информация:")
print(rest_price.info())


# In[ ]:





# In[5]:


# Ищем пропуски
print("Пропущенные значения в rest_info:")
print(rest_info.isnull().sum())

print("\nПропущенные значения в rest_price:")
print(rest_price.isnull().sum())


# ### Подготовка единого датафрейма
#

# In[4]:


# Объединение датасетов
combined_data = pd.merge(rest_info, rest_price, on='id', how='outer')

print("\n=== Объединённый датасет ===")
print("Размер объединённого датасета:", combined_data.shape)
print("\nОбщая информация об объединённом датасете:")
print(combined_data.info())
print("\nПропуски в объединённом датасете:")
print(combined_data.isnull().sum())



# ## . Предобработка данных
# 
#

# **Анализ и преобразование типов**

# 1. **rating** — числовой тип (**float**)
# 2. **chain** —  **boolean**
# 3. **seats** — количество посадочных мест (целое число)
# 4. **price** — категориальный тип (строка)
# 5. **avg_bill** — строка с описанием цен
# 6. **middle_avg_bil**l — числовой (средний чек)
# 7. **middle_coffee_cup** — числовой (цена капучино)
# 8. **name**, **address**, **district**, **category** — текстовые столбцы (**object**)
# 9. **hours** — строка с расписанием работы (**object**)
# 10. **id** — уникальный идентификатор (**object**)

# In[5]:


# Проверка результата
print("\n=== Типы данных ===")
print(combined_data.dtypes)


# In[6]:


print("\n=== Статистика по числовым столбцам ===")
print(combined_data[['rating', 'seats', 'middle_avg_bill', 'middle_coffee_cup']].describe())


# In[ ]:





# In[ ]:




# In[7]:


# Подсчёт количества пропусков
missing_counts = combined_data.isna().sum()

# Расчёт доли пропусков в процентах
missing_percentage = (missing_counts / len(combined_data)) * 100

# Объединяем в один DataFrame для удобства
missing_info = pd.DataFrame({
    'Количество пропусков': missing_counts,
    'Доля пропусков, %': missing_percentage.round(2)
})

print("=== Анализ пропущенных значений ===")
print(missing_info)


    

# In[8]:


# Выводим строки с пропусками в ключевых столбцах
print("\n=== Примеры строк с пропусками в rating ===")
print(combined_data[combined_data['rating'].isna()][['name', 'category', 'district', 'chain']].head(5))

print("\n=== Примеры строк с пропусками в seats ===")
print(combined_data[combined_data['seats'].isna()][['name', 'category', 'district']].head(5))

print("\n=== Примеры строк с пропусками в middle_avg_bill ===")
print(combined_data[combined_data['middle_avg_bill'].isna()][['name', 'category', 'avg_bill']].head(5))


# In[ ]:





# **Остваляем пропуски**:

# In[9]:


print("\n=== Исходные пропуски в данных ===")
print(combined_data.isnull().sum())




# In[10]:


print("\n=== Статистика после обработки пропусков ===")
print("Медиана посадочных мест по категориям:")
print(combined_data.groupby('category')['seats'].median().sort_values(ascending=False))


print("\nРаспределение категорий цен:")
print(combined_data['price'].value_counts())


# **Проверяем данные на дубликаты и нормализуем их**

# In[11]:



def normalize_text(text):
    if pd.isna(text):
        return text
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Нормализация текстовых столбцов
combined_data['name_normalized'] = combined_data['name'].apply(normalize_text)
combined_data['address_normalized'] = combined_data['address'].apply(normalize_text)

# Поиск скрытых дубликатов по названию и адресу
hidden_duplicates = combined_data[
    combined_data.duplicated(subset=['name_normalized', 'address_normalized'], keep=False)
]

print(f"Найдено потенциальных дубликатов (название + адрес): {len(hidden_duplicates)}")

# АНАЛИЗ ДУБЛИКАТОВ ПЕРЕД УДАЛЕНИЕМ
if len(hidden_duplicates) > 0:
    print("\n=== ПРИМЕРЫ ПОТЕНЦИАЛЬНЫХ ДУБЛИКАТОВ ===")
    # Выводим 5 случайных пар дубликатов для проверки
    sample_duplicates = hidden_duplicates.sample(min(10, len(hidden_duplicates)), random_state=42)
    print(sample_duplicates[['name', 'address', 'name_normalized', 'address_normalized']])

    # Статистика по дубликатам
    print(f"\nСтатистика по дубликатам:")
    print(f"Уникальных пар (название+адрес): {hidden_duplicates[['name_normalized', 'address_normalized']].drop_duplicates().shape[0]}")
    print(f"Среднее количество дубликатов на одну пару: {len(hidden_duplicates) / hidden_duplicates[['name_normalized', 'address_normalized']].drop_duplicates().shape[0]:.2f}")

# Поиск неявных дубликатов по названиям (сходство ≥ 85 %)
def find_fuzzy_duplicates(df, column, threshold=85):
    duplicates = []
    values = df[column].dropna().unique()
    for i, val1 in enumerate(values):
        for j, val2 in enumerate(values[i+1:], start=i+1):
            similarity = fuzz.ratio(val1, val2)
            if similarity >= threshold:
                duplicates.append({'value1': val1, 'value2': val2, 'similarity': similarity})
    return pd.DataFrame(duplicates)

fuzzy_duplicates = find_fuzzy_duplicates(combined_data, 'name_normalized', threshold=85)
print(f"\nНайдено неявных дубликатов по названиям (сходство ≥ 85 %): {len(fuzzy_duplicates)}")

if len(fuzzy_duplicates) > 0:
    print("\n=== ПРИМЕРЫ НЕЯВНЫХ ДУБЛИКАТОВ (ТОП-5 ПО СХОДСТВУ) ===")
    print(fuzzy_duplicates.sort_values('similarity', ascending=False).head())

# Удаление дубликатов только после подтверждения анализа
if len(hidden_duplicates) > 0:
    # Запрос подтверждения у пользователя (в интерактивном режиме)
    user_input = input("\nУдалить найденные дубликаты? (y/n): ").strip().lower()
    if user_input == 'y':
        cleaned_data = combined_data.drop_duplicates(
            subset=['name_normalized', 'address_normalized'],
            keep='first'
        ).reset_index(drop=True)
        print(f"Размер датасета после удаления дубликатов: {cleaned_data.shape}")
    else:
        print("Удаление отменено. Оставляем исходный датасет.")
        cleaned_data = combined_data.copy()
else:
    cleaned_data = combined_data.copy()
    print("Дубликатов не обнаружено, датасет без изменений.")

# Удаляем вспомогательные столбцы
cleaned_data = cleaned_data.drop(['name_normalized', 'address_normalized'], axis=1, errors='ignore')



# In[12]:


# Создаём новый чтолбец


def is_24_7_check(hours_text):
    """
    Проверяет, работает ли заведение 24/7.
    Возвращает True, если есть признаки круглосуточной работы 7 дней в неделю, иначе False.
    """
    if pd.isna(hours_text):
        return False

    text = str(hours_text).lower()

    # Паттерны для круглосуточной работы
    round_the_clock_patterns = [
        r'24\s*/\s*7',
        r'круглосуточно',
        r'24\s*часа'
    ]

    # Паттерны для работы 7 дней в неделю
    seven_days_patterns = [
        r'7\s*дней\s*в\s*неделю',
        r'ежедневно',
        r'каждый\s*день'
    ]

    # Проверяем наличие хотя бы одного паттерна из каждой группы
    has_round_the_clock = any(re.search(pattern, text) for pattern in round_the_clock_patterns)
    has_seven_days = any(re.search(pattern, text) for pattern in seven_days_patterns)

    return has_round_the_clock and has_seven_days

# Векторизованная версия функции
def create_is_24_7_vectorized(df):
    """Векторизованная версия создания столбца is_24_7"""
    # Создаём маску для пропусков
    mask_na = df['hours'].isna()

    # Приводим к строкам и нижнему регистру
    hours_lower = df['hours'].astype(str).str.lower()

    # Ищем паттерны для круглосуточной работы
    round_the_clock_mask = (
        hours_lower.str.contains(r'24\s*/\s*7', na=False) |
        hours_lower.str.contains(r'круглосуточно', na=False) |
        hours_lower.str.contains(r'24\s*часа', na=False)
    )

    # Ищем паттерны для 7 дней в неделю
    seven_days_mask = (
        hours_lower.str.contains(r'7\s*дней\s*в\s*неделю', na=False) |
        hours_lower.str.contains(r'ежедневно', na=False) |
        hours_lower.str.contains(r'каждый\s*день', na=False)
    )

    # Объединяем условия: и круглосуточная, и 7 дней
    is_24_7 = round_the_clock_mask & seven_days_mask

    # Устанавливаем False для пропусков
    is_24_7[mask_na] = False

    return is_24_7

# Создаём столбец is_24_7 — выбираем один из вариантов:

# Вариант 1: оптимизированный apply (если датасет небольшой)
# combined_data['is_24_7'] = combined_data['hours'].apply(is_24_7_check)


# Вариант 2: векторизованная операция (рекомендуется для больших данных)
combined_data['is_24_7'] = create_is_24_7_vectorized(combined_data)

# Проверяем результат
print("=== Статистика по столбцу is_24_7 ===")
print(combined_data['is_24_7'].value_counts())

print("\nПримеры заведений с is_24_7 = True:")
print(combined_data[combined_data['is_24_7']][['name', 'hours']].head(10))

# Дополнительная проверка: сравниваем с предыдущим подходом
print("\n=== СРАВНЕНИЕ ПОДХОДОВ ===")
# Если нужно сравнить с предыдущей версией:
# old_result = combined_data['hours'].apply(old_is_24_7_check)  # старая функция
# diff = combined_data['is_24_7'] != old_result
# print(f"Количество различий с предыдущим подходом: {diff.sum()}")




# ## . Исследовательский анализ данных


# In[13]:


# Анализ уникальных категорий и подсчёт количества объектов

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Подсчёт количества заведений по категориям
category_counts = combined_data['category'].value_counts()

print("Количество уникальных категорий:", len(category_counts))
print("\nТоп‑10 категорий по количеству заведений:")
print(category_counts.head(10))


# **Визуализация распределения категорий**
# 
# - Используем столбчатую диаграмму — она оптимально подходит для сравнения количества объектов между разными категориями.

# In[14]:


# Настройка стиля графиков
plt.style.use('default')
plt.figure(figsize=(12, 8))

# Создаём график
ax = sns.barplot(
    x=category_counts.values,
    y=category_counts.index,
    palette='viridis'
)

# Оформление графика
plt.title('Распределение заведений по категориям', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Количество заведений', fontsize=12)
plt.ylabel('Категория', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Добавляем числовые значения на столбцы
for i, v in enumerate(category_counts.values):
    ax.text(v + max(category_counts.values) * 0.01, i, str(v),
             color='black', va='center', fontsize=10)

plt.tight_layout()
plt.show()


# **Анализизуем доли категорий с помощью круговой диаграммы**

# In[15]:


# Берём топ‑15 категорий для лучшей читаемости
top_categories = category_counts.head(15)
other_count = category_counts[15:].sum()
if other_count > 0:
    top_categories['Прочие'] = other_count

plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    top_categories.values,
    labels=top_categories.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Set3.colors[:len(top_categories)]
)

plt.title('Доля категорий заведений (топ‑15 + прочие)', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')  # Для идеального круга
plt.show()


# **Построим гистограмму, чтобы увидеть, как распределяются категории по количеству заведений.**

# In[16]:


plt.figure(figsize=(10, 6))
plt.hist(category_counts.values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Распределение количества заведений по категориям', fontsize=14, fontweight='bold')
plt.xlabel('Количество заведений в категории', fontsize=12)
plt.ylabel('Частота (количество категорий)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()


# In[17]:


# Проведём анализ распределения заведений по районам, с акцентом на Центральный административный округ (ЦАО).

# Подсчёт количества заведений по районам
district_counts = combined_data['district'].value_counts()

print("Количество уникальных административных районов:", len(district_counts))
print("\nРаспределение заведений по районам (топ‑10):")
print(district_counts.head(10))


# **Используем столбчатую диаграмму для сравнения количества заведений между районами.**

# In[18]:


plt.figure(figsize=(14, 8))
ax = sns.barplot(
    x=district_counts.values,
    y=district_counts.index,
    palette='coolwarm'
)

plt.title('Распределение заведений по административным районам Москвы',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Количество заведений', fontsize=12)
plt.ylabel('Административный район', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Добавляем числовые значения на столбцы
for i, v in enumerate(district_counts.values):
    ax.text(v + max(district_counts.values) * 0.005, i, str(v),
             color='black', va='center', fontsize=9)

plt.tight_layout()
plt.show()




# **Выделим заведения в ЦАО и проанализируем их распределение по категориям.**

# In[19]:


# Фильтруем заведения в ЦАО
cao_data = combined_data[combined_data['district'] == 'Центральный административный округ']

# Подсчёт количества заведений каждой категории в ЦАО
cao_category_counts = cao_data['category'].value_counts()

print("Количество категорий заведений в ЦАО:", len(cao_category_counts))
print("\nРаспределение категорий в ЦАО (топ‑10):")
print(cao_category_counts.head(10))


# **Используем столбчатую диаграмму для наглядного сравнения.**

# In[20]:


plt.figure(figsize=(12, 8))
ax_cao = sns.barplot(
    x=cao_category_counts.values,
    y=cao_category_counts.index,
    palette='viridis'
)

plt.title('Распределение категорий заведений в Центральном административном округе',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Количество заведений', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Добавляем числовые значения
for i, v in enumerate(cao_category_counts.values):
    ax_cao.text(v + max(cao_category_counts.values) * 0.01, i, str(v),
                color='black', va='center', fontsize=10)

plt.tight_layout()
plt.show()




# **Построим круговую диаграмму, чтобы показать долю ЦАО среди всех районов.**
# 

# In[21]:


other_districts_count = len(combined_data[combined_data['district'] != 'Центральный административный округ'])
cao_count = len(cao_data)

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    [cao_count, other_districts_count],
    labels=['ЦАО', 'Остальные районы'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['#ff9999', '#66b3ff']
)

plt.title('Доля Центрального административного округа в общем количестве заведений',
          fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')
plt.show()



# **Создадим группированную столбчатую диаграмму для сравнения категорий в топ‑5 районах.**

# In[22]:


# Создаём признак: ЦАО vs остальная Москва
combined_data['region_group'] = combined_data['district'].apply(
    lambda x: 'Центральный АО' if x == 'Центральный административный округ' else 'Остальная Москва'
)

# Сводная таблица с долями по категориям (в %)
comparison_pct = pd.crosstab(
    combined_data['region_group'],
    combined_data['category'],
    normalize='index'  # Нормализуем по строкам — получаем доли
) * 100  # Переводим в проценты

# Визуализация: группированная столбчатая диаграмма
fig, ax = plt.subplots(figsize=(14, 8))

# Строим столбцы для каждой категории рядом: один для ЦАО, другой для остальной Москвы
comparison_pct.T.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff'], width=0.8)

plt.title('Сравнение долей категорий заведений: ЦАО vs Остальная Москва\n(в процентах от общего числа заведений в группе)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Категория заведения', fontsize=12)
plt.ylabel('Доля в %', fontsize=12)
plt.legend(title='Регион', labels=['Центральный АО', 'Остальная Москва'],
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем процентные значения на столбцы (альтернатива bar_label)
for i, patch in enumerate(ax.patches):
    # Получаем высоту столбца (значение процента)
    height = patch.get_height()
    if not pd.isna(height) and height > 0:  # Проверяем, что значение есть и положительное
        ax.text(
            patch.get_x() + patch.get_width() / 2,  # Координата X: центр столбца
            height + 0.5,  # Координата Y: чуть выше вершины столбца
            f'{height:.1f}%',  # Текст: процент с одним знаком после запятой
            ha='center', va='bottom', fontsize=9, color='black'
        )

plt.tight_layout()
plt.show()

# Вывод таблицы с данными для проверки
print("Доли категорий заведений (в %):")
print(comparison_pct.round(1))




# **Проведём анализ соотношения сетевых и несетевых заведений в целом и по категориям.**

# In[23]:


# Подсчёт общего количества сетевых и несетевых заведений


chain_counts = combined_data['chain'].value_counts()
chain_labels = ['Несетевые', 'Сетевые']

print("Общее распределение:")
print(f"Несетевые: {chain_counts[0]} ({chain_counts[0]/len(combined_data)*100:.1f}%)")
print(f"Сетевые: {chain_counts[1]} ({chain_counts[1]/len(combined_data)*100:.1f}%)")


# **Используем круговую диаграмму — она наглядно показывает доли категорий.**

# In[24]:


plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    chain_counts.values,
    labels=chain_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#ff9999', '#66b3ff'],
    explode=[0.03, 0]  # Небольшой акцент на сетевых
)

plt.title('Соотношение сетевых и несетевых заведений\n(все данные)',
          fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.show()


# **Создаём сводную таблицу распределения сетевых/несетевых заведений по категориям.**

# In[25]:


# Сводная таблица: категории × сетевые/несетевые
category_chain = pd.crosstab(
    combined_data['category'],
    combined_data['chain'],
    margins=True,
    margins_name='Всего'
)

# Переименовываем столбцы для понятности
category_chain.columns = ['Несетевые', 'Сетевые', 'Всего']

# Добавляем процент сетевых заведений по каждой категории
category_chain['% сетевых'] = (
    category_chain['Сетевые'] / category_chain['Всего'] * 100
).round(1)

print("Распределение по категориям (топ‑10 по количеству заведений):")
print(category_chain.sort_values('Всего', ascending=False).head(10))


# **Используем столбчатую диаграмму с группировкой — она позволяет сравнить количество сетевых и несетевых заведений внутри каждой категории.**

# In[26]:


# Берём топ‑15 категорий по общему количеству заведений
top_categories = category_chain.sort_values('Всего', ascending=False).head(15)

plt.figure(figsize=(14, 8))
x = range(len(top_categories.index))
width = 0.35

plt.bar(x, top_categories['Несетевые'], width, label='Несетевые', color='#ff9999')
plt.bar([i + width for i in x], top_categories['Сетевые'], width, label='Сетевые', color='#66b3ff')

plt.xlabel('Категория заведения', fontsize=12)
plt.ylabel('Количество заведений', fontsize=12)
plt.title('Распределение сетевых и несетевых заведений по категориям\n(топ‑15 категорий)',
          fontsize=14, fontweight='bold', pad=20)
plt.xticks([i + width/2 for i in x], top_categories.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# **Построим столбчатую диаграмму, показывающую процент сетевых заведений в каждой категории — это поможет выявить наиболее «сетевые» форматы.**

# In[27]:


# Сортируем по проценту сетевых заведений (убывание)
top_categories_sorted = top_categories.sort_values('% сетевых', ascending=False)

plt.figure(figsize=(12, 7))
bars = plt.barh(
    top_categories_sorted.index,
    top_categories_sorted['% сетевых'],
    color=plt.cm.viridis(top_categories_sorted['% сетевых']/100)
)

plt.xlabel('Доля сетевых заведений, %', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)
plt.title('Доля сетевых заведений по категориям\n(топ‑15 категорий по количеству заведений)',
          fontsize=14, fontweight='bold', pad=20)

# Добавляем числовые значения на столбцы
for bar, percent in zip(bars, top_categories_sorted['% сетевых']):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
              f'{percent:.1f}%', ha='left', va='center', fontsize=9)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()



# In[30]:


# Первичный анализ распределения посадочных мест

# Удаляем NaN и преобразуем к числовому типу
seats_clean = pd.to_numeric(combined_data['seats'], errors='coerce').dropna()

print("Статистика по количеству посадочных мест:")
print(seats_clean.describe())

# Визуализация распределения
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(seats_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Распределение посадочных мест\n(все заведения)', fontsize=14, fontweight='bold')
plt.xlabel('Количество посадочных мест', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(seats_clean, vert=False)
plt.title('Ящик с усами: распределение посадочных мест', fontsize=14, fontweight='bold')
plt.xlabel('Количество посадочных мест', fontsize=12)
plt.tight_layout()
plt.show()


# In[31]:


# Используем межквартильный размах (IQR) для обнаружения аномальных значений.

Q1 = seats_clean.quantile(0.25)
Q3 = seats_clean.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = seats_clean[(seats_clean < lower_bound) | (seats_clean > upper_bound)]
print(f"\nОбнаружено выбросов: {len(outliers)}")
print(f"Границы выбросов: от {lower_bound:.1f} до {upper_bound:.1f}")
print(f"Максимальные выбросы: {outliers.sort_values(ascending=False).head(10).values}")


# In[32]:


# Исследуем, в каких категориях встречаются аномальные значения.

# Создаём столбец с меткой выброса
combined_data['is_outlier'] = False
seats_numeric = pd.to_numeric(combined_data['seats'], errors='coerce')
combined_data.loc[seats_numeric.notna(), 'is_outlier'] = (
    (seats_numeric[seats_numeric.notna()] < lower_bound) |
    (seats_numeric[seats_numeric.notna()] > upper_bound)
)

# Группируем по категориям и считаем выбросы
outlier_by_category = combined_data.groupby('category')['is_outlier'].sum().sort_values(ascending=False)
print("\nКоличество выбросов по категориям:")
print(outlier_by_category[outlier_by_category > 0])


# In[33]:


# Построим боксплоты для каждой категории — это позволит увидеть распределение и выбросы в разрезе категорий.

plt.figure(figsize=(16, 10))
# Берём топ‑15 категорий по количеству заведений
top_categories = combined_data['category'].value_counts().head(15).index
data_for_boxplot = combined_data[combined_data['category'].isin(top_categories)]

sns.boxplot(
    data=data_for_boxplot,
    x='seats',
    y='category',
    palette='Set3'
)
plt.title('Распределение посадочных мест по категориям заведений\n(с указанием выбросов)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Количество посадочных мест', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[34]:


# Рассчитаем медиану для каждой категории.

typical_seats = combined_data.groupby('category')['seats'].agg([
    'count', 'mean', 'median', 'std'
]).round(2).sort_values('median', ascending=False)

print("Типичное количество посадочных мест по категориям (по медиане):")
print(typical_seats[['count', 'median', 'mean', 'std']])


# In[35]:


# Создадим столбчатую диаграмму с медианами по категориям.

plt.figure(figsize=(14, 8))
ax = sns.barplot(
    data=typical_seats.reset_index().head(15),
    x='median',
    y='category',
    palette='viridis'
)
plt.title('Типичное количество посадочных мест (медиана) по категориям\n(топ‑15 категорий)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Медиана количества посадочных мест', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)

# Добавляем числовые значения
for i, v in enumerate(typical_seats['median'].head(15)):
    ax.text(v + max(typical_seats['median']) * 0.01, i, f'{v:.0f}',
             color='black', va='center', fontsize=10)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()




# In[36]:


# Группируем по категории и рассчитываем средний рейтинг
category_ratings = combined_data.groupby('category')['rating'].agg([
    'mean', 'median', 'count', 'std'
]).round(3).sort_values('mean', ascending=False)

print("Средние рейтинги по категориям заведений:")
print(category_ratings)


# In[37]:


# Визуализация средних рейтингов (используем столбчатую диаграмму)


plt.figure(figsize=(14, 8))

# Создаём столбчатую диаграмму
bars = plt.barh(
    category_ratings.index,
    category_ratings['mean'],
    color='skyblue',
    edgecolor='navy',
    linewidth=0.8,
    alpha=0.8
)

# Добавляем значения на столбцы
for i, v in enumerate(category_ratings['mean']):
    plt.text(v + 0.01, i, f'{v:.2f}',
             color='darkblue', va='center', fontsize=10)

plt.xlabel('Средний рейтинг (по 5‑балльной шкале)', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)
plt.title('Распределение средних рейтингов по категориям заведений',
          fontsize=16, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[38]:


# Детальная визуализация распределения
# Построим боксплоты для каждой категории — это покажет не только среднее, но и разброс оценок.


plt.figure(figsize=(16, 10))
sns.boxplot(
    data=combined_data,
    x='rating',
    y='category',
    palette='Set3',
    showfliers=False  # Убираем выбросы для лучшей читаемости
)
plt.title('Распределение рейтингов по категориям заведений\n(без учёта выбросов)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Рейтинг заведения', fontsize=12)
plt.ylabel('Категория заведения', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[39]:


# Дополнительная визуализация: количество заведений в каждой категории


plt.figure(figsize=(14, 6))
category_counts = combined_data['category'].value_counts()

bars = plt.bar(
    category_counts.index,
    category_counts.values,
    color='lightgreen',
    edgecolor='darkgreen',
    alpha=0.8
)

# Добавляем подписи на столбцы
for i, v in enumerate(category_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Количество заведений', fontsize=12)
plt.title('Количество заведений по категориям',
          fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:





# In[53]:


# Создаём копию данных для анализа
data_for_corr = combined_data.copy()

# Определяем столбцы для анализа
analysis_columns = ['rating', 'category', 'district', 'chain', 'is_24_7', 'seats', 'price']

# Фильтруем только существующие столбцы
available_columns = [col for col in analysis_columns if col in data_for_corr.columns]
corr_data = data_for_corr[available_columns].copy()

print(f"Столбцы для корреляционного анализа: {available_columns}")

# Удаляем строки с пропусками в ключевом столбце (рейтинг)
corr_data = corr_data[corr_data['rating'].notna()]

# Указываем, какие столбцы являются интервальными (непрерывными количественными)
# price — категориальный, не включаем его в интервальные
interval_columns = ['seats', 'rating']
valid_interval_cols = [col for col in interval_columns if col in corr_data.columns]

# Преобразуем price в категориальный тип (явное указание)
if 'price' in corr_data.columns:
    corr_data['price'] = corr_data['price'].astype('category')

# Рассчитываем матрицу корреляции phik
if valid_interval_cols:
    correlation_matrix = corr_data.phik_matrix(
        interval_cols=valid_interval_cols,
        bins=10,
        dropna=True   # Игнорируем пропуски
    )
else:
    # Если нет интервальных столбцов, рассчитываем без указания interval_cols
    correlation_matrix = corr_data.phik_matrix(dropna=True)

print(f"\nФинальный размер данных для корреляции: {corr_data.shape}")
print("\nМатрица корреляции phik (финальный расчёт):")
print(correlation_matrix.round(3))

# Дополнительно: выводим статистику по пропускам
print("\nСтатистика пропусков в данных для анализа:")
print(corr_data.isnull().sum())

# Выводим типы данных для проверки
print("\nТипы данных после преобразований:")
print(corr_data.dtypes)



# In[52]:


print("Информация о данных для корреляции:")
print(corr_data.info())
print("\nПропущенные значения в каждом столбце:")
print(corr_data.isnull().sum())
print("\nПервые 5 строк данных:")
print(corr_data.head())


# **Строим матрицу кореляции**

# In[54]:


# Определяем столбцы
interval_columns = ['rating', 'category_encoded', 'district_encoded', 'seats_numeric']

# Расчёт матрицы корреляций phik
phik_correlation_matrix = phik.phik_matrix(
    corr_data,
    interval_cols=interval_columns,
    bins=10
)

# Визуализация
plt.figure(figsize=(12, 10))
sns.heatmap(
    phik_correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    fmt='.3f',
    cbar_kws={'shrink': 0.8},
    linewidths=0.5
)
plt.title('Матрица корреляций phik: рейтинг заведения и различные факторы',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Вывод в текстовом виде
print("Матрица корреляций phik:")
print(phik_correlation_matrix.round(3))


#  
#     
# In[55]:


# Найдём самую сильную связь с рейтингом:

# анализ общей матрицы phik
rating_correlations = correlation_matrix['rating'].abs().sort_values(ascending=False)
print("Корреляция рейтинга с различными параметрами (по убыванию силы связи):")
print(rating_correlations)

strongest_corr_feature = rating_correlations.index[1]
strongest_corr_value = rating_correlations[strongest_corr_feature]
strongest_actual_corr = correlation_matrix['rating'][strongest_corr_feature]

print(f"\nСамая сильная связь: {strongest_corr_feature}")
print(f"Коэффициент корреляции (phik): {strongest_actual_corr:.3f}")
print(f"Абсолютное значение: {strongest_corr_value:.3f}")


if strongest_corr_feature == 'price':
    print("\n Самая сильная связь действительно с ценой. Проводим углублённый анализ...")
else:
    print(f"\n Самая сильная связь не с ценой, а с '{strongest_corr_feature}'. Тем не менее, проанализируем связь 'цена ↔ рейтинг' отдельно:")



# In[56]:


# Проверка самой сильной связи 


if 'category_encoded' in correlation_matrix.columns:
    # Группируем по цене и считаем средний рейтинг
    price_rating = corr_data.groupby('category_encoded')['rating'].mean().reset_index()

    # Определяем порядок категорий (если есть логический порядок)
    price_order = ['Низкий', 'Средний', 'Высокий']  # замените на ваши категории
    if 'category_encoded' in corr_data.columns and corr_data['category_encoded'].dtype == 'object':
        price_rating['category_encoded'] = pd.Categorical(
            price_rating['category_encoded'],
            categories=price_order,
            ordered=True
        )
        price_rating = price_rating.sort_values('category_encoded')

    # Построение графика
    plt.figure(figsize=(10, 6))
    sns.barplot(data=price_rating, x='category_encoded', y='rating', palette='Blues_d')
    plt.title('Средний рейтинг заведения по уровням цен', fontsize=16, fontweight='bold')
    plt.xlabel('Уровень цены', fontsize=12)
    plt.ylabel('Средний рейтинг', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
else:
    print("Столбец 'price' отсутствует в матрице корреляций.")


# In[57]:


print("АНАЛИЗ САМОЙ СИЛЬНОЙ СВЯЗИ")
print(f"Признак: {strongest_corr_feature}")
print(f"Коэффициент корреляции (phik): {strongest_actual_corr:.3f}")

# Интерпретация
if strongest_actual_corr >= 0.7:
    strength = "сильная"
elif strongest_actual_corr >= 0.5:
    strength = "умеренная"
elif strongest_actual_corr >= 0.3:
    strength = "заметная"
elif strongest_actual_corr >= 0.1:
    strength = "слабая"
else:
    strength = "практически отсутствует"

print(f"Интерпретация: связь {strength}")
print("Вывод: тип заведения (категория) практически не влияет на его рейтинг.")


# In[58]:


# анализ общей матрицы phik
rating_correlations = correlation_matrix['rating'].abs().sort_values(ascending=False)
print("Корреляция рейтинга с различными параметрами (по убыванию силы связи):")
print(rating_correlations)

strongest_corr_feature = rating_correlations.index[1]
strongest_corr_value = rating_correlations[strongest_corr_feature]
strongest_actual_corr = correlation_matrix['rating'][strongest_corr_feature]

print(f"\nСамая сильная связь: {strongest_corr_feature}")
print(f"Коэффициент корреляции (phik): {strongest_actual_corr:.3f}")
print(f"Абсолютное значение: {strongest_corr_value:.3f}")

# Анализ самой сильной связи
if strongest_corr_feature == 'price':
    print("\n Самая сильная связь действительно с ценой. Проводим углублённый анализ...")
else:
    print(f"\n Самая сильная связь не с ценой, а с '{strongest_corr_feature}'. Тем не менее, проанализируем связь 'цена ↔ рейтинг' отдельно:")

# Выполняем анализ самой сильной связи (шаг 1) и проверку наличия price (шаг 2)
# Затем — углублённый анализ (шаг 3), если price найден


# In[59]:


# Анализ самой сильной связи (на основе матрицы phik)
print("АНАЛИЗ САМОЙ СИЛЬНОЙ СВЯЗИ С РЕЙТИНГОМ")
rating_correlations = phik_correlation_matrix['rating'].abs().sort_values(ascending=False)
print("Корреляция рейтинга с различными параметрами (по убыванию силы связи):")
print(rating_correlations)

strongest_corr_feature = rating_correlations.index[1]
strongest_corr_value = rating_correlations[strongest_corr_feature]
strongest_actual_corr = phik_correlation_matrix['rating'][strongest_corr_feature]

print(f"\nСамая сильная связь: {strongest_corr_feature}")
print(f"Коэффициент корреляции (phik): {strongest_actual_corr:.3f}")
print(f"Абсолютное значение: {strongest_corr_value:.3f}")

# Интерпретация силы связи
if strongest_actual_corr >= 0.7:
    strength = "сильная"
elif strongest_actual_corr >= 0.5:
    strength = "умеренная"
elif strongest_actual_corr >= 0.3:
    strength = "заметная"
elif strongest_actual_corr >= 0.1:
    strength = "слабая"
else:
    strength = "практически отсутствует"

print(f"Интерпретация: связь {strength}")
print(f"Вывод: {strongest_corr_feature} практически {'не влияет' if strength == 'практически отсутствует' else 'влияет'} на рейтинг заведения.")


# **пришлось делать проверку столбца price , так как без него выдаёт ошибку**

# In[65]:


# Проверка наличия столбца price в исходных данных
print("\nПРОВЕРКА НАЛИЧИЯ СТОЛБЦА 'price' В ИСХОДНЫХ ДАННЫХ:")
if 'price' in combined_data.columns:
    print(" Столбец 'price' найден в исходных данных")
    price_null_count = combined_data['price'].isnull().sum()
    print(f"Пропущенные значения в 'price': {price_null_count} из {len(combined_data)}")

    # Проверяем, есть ли хоть какие‑то заполненные значения
    if price_null_count < len(combined_data):
        print("Столбец содержит данные — можно провести углублённый анализ")
        price_available = True
    else:
        print("Столбец полностью пустой — углублённый анализ невозможен")
        price_available = False
else:
    print(" Столбец 'price' отсутствует в исходных данных")
    price_available = False


# In[110]:


print("\nДЕТАЛЬНЫЙ АНАЛИЗ ДАННЫХ ДЛЯ КОРРЕЛЯЦИИ 'ЦЕНА ↔ РЕЙТИНГ':")

# Создаём временный DataFrame с нужными столбцами
analysis_data = combined_data[['rating', 'price']].copy()

# Удаляем строки с пропусками в rating (в нашем случае их нет, но оставляем для общности)
analysis_data = analysis_data[analysis_data['rating'].notna()]
print(f"[ДИАГНОСТИКА] Строк после удаления пропусков в rating: {len(analysis_data)}")

if len(analysis_data) == 0:
    print(" Нет данных с заполненным 'rating' — анализ невозможен")
else:
    # Создаём mapping для приведения к стандартным категориям
    price_mapping = {
        'низкие': 'Низкий',
        'средние': 'Средний',
        'выше среднего': 'Выше среднего',
        'высокие': 'Высокий'
    }

    # Применяем mapping
    analysis_data['price'] = analysis_data['price'].map(price_mapping).fillna(analysis_data['price'])

    # Задаём явный порядок категорий (важно для корреляции)
    price_order = ['Низкий', 'Средний', 'Выше среднего', 'Высокий']
    analysis_data['price'] = pd.Categorical(
        analysis_data['price'],
        categories=price_order,
        ordered=True
    )

    # Удаляем пропуски в price
    valid_data = analysis_data.dropna(subset=['price'])
    print(f"[ДИАГНОСТИКА] Строк с заполненными 'rating' и 'price' после перекодировки: {len(valid_data)}")

    if len(valid_data) < 2:
        print(" Недостаточно данных для расчёта корреляции")
    else:
        # Визуализация среднего рейтинга по уровням цены
        price_rating = valid_data.groupby('price')['rating'].agg(['count', 'mean', 'std']).round(3)
        print(f"\n[ДИАГНОСТИКА] Статистика по уровням цены:")
        print(price_rating)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=price_rating.reset_index(),
            x='price',
            y='mean',
            palette='Blues_d'
        )
        plt.title('Средний рейтинг заведения по уровням цен', fontsize=16, fontweight='bold')
        plt.xlabel('Уровень цены', fontsize=12)
        plt.ylabel('Средний рейтинг', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        # Подписи с количеством наблюдений
        for i, row in enumerate(price_rating.reset_index().itertuples()):
            plt.text(i, row.mean + 0.01, f"n={row.count}", ha='center', fontsize=9)
        plt.tight_layout()
        plt.show()

        # Расчёт корреляции Кендалла
        valid_data['price_code'] = valid_data['price'].cat.codes
        corr_matrix = valid_data[['rating', 'price_code']].corr(method='kendall')
        tau = corr_matrix.loc['rating', 'price_code']
        print(f"\nКОРРЕЛЯЦИЯ КЕНДАЛЛА МЕЖДУ ЦЕНОЙ И РЕЙТИНГОМ:")
        print(f"τ = {tau:.3f}")

        # Интерпретация результата
        if abs(tau) < 0.1:
            strength = "отсутствует"
        elif abs(tau) < 0.3:
            strength = "слабая"
        elif abs(tau) < 0.5:
            strength = "заметная"
        elif abs(tau) < 0.7:
            strength = "умеренная"
        else:
            strength = "сильная"

        direction = "положительная" if tau > 0 else "отрицательная" if tau < 0 else "нет направления"
        print(f"Сила: {strength}, направление: {direction}")

        # Дополнительный вывод
        if tau > 0:
            print(f"Вывод: с ростом уровня цены средний рейтинг заведения {'увеличивается' if tau > 0.1 else 'незначительно растёт'}")
        elif tau < 0:
            print(f"Вывод: с ростом уровня цены средний рейтинг заведения {'снижается' if abs(tau) > 0.1 else 'незначительно снижается'}")
        else:
            print("Вывод: между уровнем цены и рейтингом нет заметной связи")



# **Выполним анализ сетевых заведений: найдём топ‑15 по количеству точек, рассчитаем их средние рейтинги и определим категории.**

# In[111]:


# Фильтруем только сетевые заведения (chain == 1)
chain_restaurants = combined_data[combined_data['chain'] == 1].copy()


print(f"Найдено сетевых заведений: {len(chain_restaurants)}")

if len(chain_restaurants) == 0:
    print("Нет сетевых заведений в данных. Проверьте кодировку столбца 'chain'.")
else:
    # Группируем по названию сети и считаем количество заведений
    chain_stats = chain_restaurants.groupby('name').agg({
        'rating': 'mean',
        'category': 'first',  # Берём первую попавшуюся категорию для сети
        'id': 'count'  # Количество заведений
    }).round(3)

    # Переименовываем столбцы
    chain_stats.columns = ['avg_rating', 'category', 'count']
    chain_stats = chain_stats.reset_index()

    # Сортируем по количеству заведений (по убыванию) и берём топ‑15
    top_15_chains = chain_stats.sort_values('count', ascending=False).head(15)

    print("\nТоп‑15 популярных сетей заведений в Москве:")
    print(top_15_chains)


# In[112]:


# Визуализируем топ‑15 сетей по количеству заведений

plt.figure(figsize=(14, 8))

# Создаём столбчатую диаграмму
bars = plt.barh(
    top_15_chains['name'],
    top_15_chains['count'],
    color='lightcoral',
    edgecolor='darkred',
    alpha=0.8
)

# Добавляем значения на столбцы
for i, v in enumerate(top_15_chains['count']):
    plt.text(v + 0.5, i, str(v), color='darkred', va='center', fontsize=10)

plt.xlabel('Количество заведений в сети', fontsize=12)
plt.ylabel('Название сети', fontsize=12)
plt.title('Топ‑15 самых популярных сетей заведений в Москве (по количеству точек)',
          fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()  # Чтобы самая популярная сеть была сверху
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[113]:


plt.figure(figsize=(14, 8))

# Столбчатая диаграмма средних рейтингов
bars = plt.barh(
    top_15_chains['name'],
    top_15_chains['avg_rating'],
    color='skyblue',
    edgecolor='navy',
    alpha=0.8
)

# Добавляем значения рейтингов на столбцы
for i, v in enumerate(top_15_chains['avg_rating']):
    plt.text(v + 0.01, i, f'{v:.2f}', color='darkblue', va='center', fontsize=10)

plt.xlabel('Средний рейтинг (по 5‑балльной шкале)', fontsize=12)
plt.ylabel('Название сети', fontsize=12)
plt.title('Средние рейтинги топ‑15 сетей заведений',
          fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# **Посмотрим, к каким категориям относятся заведения в топ‑15 сетях.**

# In[114]:


# Считаем количество сетей по категориям
category_distribution = top_15_chains['category'].value_counts()

plt.figure(figsize=(12, 6))

# Круговая диаграмма распределения по категориям
plt.pie(
    category_distribution.values,
    labels=category_distribution.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Set3.colors
)
plt.title('Распределение топ‑15 сетей по категориям заведений',
          fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')  # Для идеального круга
plt.tight_layout()
plt.show()

print("Распределение топ‑15 сетей по категориям:")
print(category_distribution)


# In[115]:


# Создадим график, который одновременно показывает количество заведений и их средний рейтинг.


fig, ax1 = plt.subplots(figsize=(16, 8))

# Первая ось — количество заведений (столбцы)
color1 = 'lightcoral'
ax1.set_xlabel('Название сети')
ax1.set_ylabel('Количество заведений', color=color1)
bars1 = ax1.bar(
    top_15_chains['name'],
    top_15_chains['count'],
    color=color1,
    alpha=0.7,
    label='Количество заведений'
)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(axis='y', alpha=0.3)

# Вторая ось — средний рейтинг (линия)
ax2 = ax1.twinx()
color2 = 'skyblue'
ax2.set_ylabel('Средний рейтинг', color=color2)
line2 = ax2.plot(
    top_15_chains['name'],
    top_15_chains['avg_rating'],
    color=color2,
    marker='o',
    linewidth=2,
    markersize=8,
    label='Средний рейтинг'
)
ax2.tick_params(axis='y', labelcolor=color2)

# Настройка подписей и заголовка
plt.title('Топ‑15 сетей: количество заведений и средний рейтинг',
          fontsize=16, fontweight='bold', pad=20)
fig.tight_layout()

# Легенда
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Поворот подписей по оси X для лучшей читаемости
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[116]:


#Выведем сводную таблицу с основными показателями.


print("Сводная таблица по топ‑15 сетям заведений:")
summary_table = top_15_chains[['name', 'category', 'count', 'avg_rating']].copy()
summary_table.columns = ['Название сети', 'Категория', 'Количество заведений', 'Средний рейтинг']
print(summary_table.to_string(index=False))





# **Выполним анализ зависимости среднего чека от расположения заведения, с акцентом на сравнение ЦАО с другими округами.**

# In[117]:


# Копируем данные для анализа
price_analysis = combined_data.copy()

# Преобразуем средний чек в числовой формат
price_analysis['middle_avg_bill_numeric'] = pd.to_numeric(
    price_analysis['middle_avg_bill'],
    errors='coerce'
)

# Удаляем строки с пропусками и нулевыми/отрицательными чеками
price_analysis = price_analysis[
    (price_analysis['middle_avg_bill_numeric'].notna()) &
    (price_analysis['middle_avg_bill_numeric'] > 0)
]

print(f"Всего заведений с корректными данными о среднем чеке: {len(price_analysis)}")


# In[118]:



# Группируем по районам и рассчитываем статистику
district_stats = price_analysis.groupby('district').agg({
    'middle_avg_bill_numeric': ['count', 'mean', 'median', 'std', 'min', 'max']
}).round(2)

# Разворачиваем многоуровневые столбцы
district_stats.columns = [
    'count_restaurants', 'avg_bill', 'median_bill',
    'std_bill', 'min_bill', 'max_bill'
]
district_stats = district_stats.reset_index()

# Сортируем по среднему чеку (по убыванию)
district_stats_sorted = district_stats.sort_values('avg_bill', ascending=False)

print("Статистика среднего чека по районам:")
print(district_stats_sorted)


# In[119]:


# Средний чек по всем районам отобразим на столбчатой диаграмме

plt.figure(figsize=(16, 10))

# Создаём столбчатую диаграмму
bars = plt.barh(
    district_stats_sorted['district'],
    district_stats_sorted['avg_bill'],
    color='lightgreen',
    edgecolor='darkgreen',
    alpha=0.8
)

# Добавляем значения на столбцы
for i, v in enumerate(district_stats_sorted['avg_bill']):
    plt.text(v + 50, i, f'{v:.0f} ₽', color='darkgreen', va='center', fontsize=9)

plt.xlabel('Средний чек (руб.)', fontsize=12)
plt.ylabel('Район Москвы', fontsize=12)
plt.title('Средний чек заведений по районам Москвы',
          fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()  # Самая высокая цена сверху
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[120]:


# Создадим бинарную переменную «в центре/не в центре»:


# Определяем, является ли район центральным
price_analysis['is_central'] = price_analysis['district'].apply(
    lambda x: 'ЦАО' if x == 'Центральный административный округ' else 'Другие округа'
)

# Статистика для ЦАО и остальных
central_vs_others = price_analysis.groupby('is_central').agg({
    'middle_avg_bill_numeric': ['count', 'mean', 'median', 'std']
}).round(2)
central_vs_others.columns = ['count', 'avg', 'median', 'std']

print("Сравнение ЦАО с остальными округами:")
print(central_vs_others)


# In[121]:


# Визуализируем сравнения:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Боксплот для ЦАО vs другие округа
sns.boxplot(
    data=price_analysis,
    x='is_central',
    y='middle_avg_bill_numeric',
    palette='Set2',
    ax=ax1
)
ax1.set_title('Распределение среднего чека:\nЦАО vs Другие округа',
             fontsize=14, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('Средний чек (руб.)')
ax1.grid(axis='y', alpha=0.3)

# Столбчатая диаграмма средних значений
categories = ['ЦАО', 'Другие округа']
means = central_vs_others['avg'].values
bars = ax2.bar(
    categories,
    means,
    color=['red', 'blue'],
    alpha=0.7
)
for i, v in enumerate(means):
    ax2.text(i, v + 100, f'{v:.0f} ₽', ha='center', va='bottom', fontsize=11)
ax2.set_title('Средние значения чека', fontsize=14, fontweight='bold')
ax2.set_ylabel('Средний чек (руб.)')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Сравнение среднего чека в ЦАО и других округах Москвы',
           fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[122]:


# Для анализа влияния удалённости создадим упрощённую шкалу удалённости:


# Создаём категории удалённости (примерная классификация)
distance_mapping = {
    'Центральный административный округ': 'Центр (ЦАО)',
    'Северный административный округ': 'Ближний пояс',
    'Северо-Восточный административный округ': 'Ближний пояс',
    'Восточный административный округ': 'Ближний пояс',
    'Юго-Восточный административный округ': 'Ближний пояс',
    'Южный административный округ': 'Ближний пояс',
    'Юго-Западный административный округ': 'Ближний пояс',
    'Западный административный округ': 'Ближний пояс',
    'Северо-Западный административный округ': 'Ближний пояс',
    'Зеленоградский административный округ': 'Дальний пояс',
    'Троицкий административный округ': 'Дальний пояс',
    'Новомосковский административный округ': 'Дальний пояс'
}


price_analysis['distance_from_center'] = price_analysis['district'].map(distance_mapping)

# Статистика по удалённости
distance_stats = price_analysis.groupby('distance_from_center').agg({
    'middle_avg_bill_numeric': ['count', 'mean', 'median']
}).round(2)
distance_stats.columns = ['count', 'avg_bill', 'median_bill']

print("Зависимость среднего чека от удалённости от центра:")
print(distance_stats)


# In[123]:


# Визуализируем зависимости:


plt.figure(figsize=(12, 6))

# Столбчатая диаграмма по зонам удалённости
categories = distance_stats.index
means = distance_stats['avg_bill'].values
bars = plt.bar(
    categories,
    means,
    color=['red', 'orange', 'lightblue'],
    alpha=0.8,
    edgecolor='black'
)

for i, v in enumerate(means):
    plt.text(i, v + 50, f'{v:.0f} ₽', ha='center', va='bottom', fontsize=11)

plt.xlabel('Удалённость от центра', fontsize=12)
plt.ylabel('Средний чек (руб.)', fontsize=12)
plt.title('Зависимость среднего чека от удалённости от центра Москвы',
          fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

