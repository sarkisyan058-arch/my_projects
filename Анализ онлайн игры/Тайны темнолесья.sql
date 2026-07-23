/* Проект «Секреты Тёмнолесья»
 * Цель проекта: изучить влияние характеристик игроков и их игровых персонажей 
 * на покупку внутриигровой валюты «райские лепестки», а также оценить 
 * активность игроков при совершении внутриигровых покупок
 * 
 * Автор: Артем Саркисян
 * Дата: 2025-11-26

**/

-- Часть 1.1: Исследование доли платящих игроков по всем данным
SELECT
    COUNT(id) AS total_players,                          -- общее количество игроков
    SUM(payer) AS paying_players,                       -- количество платящих игроков
    ROUND(SUM(payer)::NUMERIC / (COUNT(id)::NUMERIC) * 100, 2) AS paying_ratio       -- доля платящих игроков
FROM fantasy.users;

-- Часть 1.2: Доля платящих игроков в разрезе расы
SELECT
    r.race,                                                               -- раса персонажа
    COUNT(CASE WHEN u.payer = 1 THEN u.id END) AS paying_players_race,    -- платящие игроки расы
    COUNT(u.id) AS total_players_race,                                    -- общее количество игроков расы
    COUNT(CASE WHEN u.payer = 1 THEN u.id END)::NUMERIC / COUNT(u.id) AS paying_ratio_race  -- доля платящих
FROM fantasy.users u
JOIN fantasy.race r ON u.race_id = r.race_id
GROUP BY r.race;

-- Часть 2.1: Статистика покупок
SELECT
    COUNT(*) AS total_purchases,  -- общее количество покупок
    SUM(amount) AS total_spent,  -- суммарная стоимость
    MIN(amount) AS min_amount,   -- минимальная покупка
    MAX(amount) AS max_amount,   -- максимальная покупка
    ROUND(AVG(amount)) AS avg_amount,   -- средняя стоимость
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)) AS median_amount,  -- медиана
    ROUND(STDDEV(amount)) AS stddev_amount,                                       -- стандартное отклонение
    (COUNT(DISTINCT CASE WHEN amount > 0 THEN id END) ::FLOAT / 
     NULLIF(COUNT(DISTINCT id), 0)) AS payer_share                         -- Доля платящих покупателей
FROM fantasy.events

-- Часть 2.2: Нулевые покупки
SELECT
    COUNT(*) AS zero_purchases,                          -- количество нулевых покупок
    COUNT(*)::NUMERIC / (SELECT COUNT(*) FROM fantasy.events) AS zero_ratio  -- доля нулевых
FROM fantasy.events
WHERE amount = 0;

-- Часть 2.3: Популярность предметов
SELECT
    i.game_items,                                        -- название предмета
    COUNT(*) AS sales_count,                           -- количество продаж
    ROUND(
        (COUNT(*)::NUMERIC / (
            SELECT COUNT(*) 
            FROM fantasy.events 
            WHERE amount > 0
        )) * 100,
        2
    ) AS sales_percent,                                -- доля продаж в % от всех покупок
    COUNT(DISTINCT id) AS unique_buyers,     -- уникальных покупателей предмета
    ROUND(
        (COUNT(DISTINCT id)::NUMERIC / (
            SELECT COUNT(DISTINCT id) 
            FROM fantasy.events
            WHERE amount > 0
        )) * 100,
        2
    ) AS buyer_percent                                  -- доля покупателей предмета в % от всех платящих
FROM fantasy.events e
JOIN fantasy.items i ON e.item_code = i.item_code
WHERE e.amount > 0
GROUP BY i.game_items
ORDER BY sales_count DESC;

SELECT
    r.race, --рассы
    COUNT(DISTINCT u.id) AS total_players,  --общее количество зарегистрированных игроков
    COUNT(DISTINCT CASE WHEN e.amount > 0 THEN u.id END) AS purchasers,  -- Доля покупателей среди всех игроков расы
    ROUND(
        COUNT(DISTINCT CASE WHEN e.amount > 0 THEN u.id END)::NUMERIC
        / COUNT(DISTINCT u.id) * 100,
        2
    ) AS purchaser_ratio_percent,  -- Доля платящих среди ВСЕХ игроков расы в % (согласована с Частью 1)
    ROUND(
        COUNT(DISTINCT CASE 
            WHEN u.payer = 1 AND e.amount > 0 THEN u.id
            ELSE NULL
        END)::NUMERIC
        / NULLIF(COUNT(DISTINCT CASE WHEN e.amount > 0 THEN u.id END), 0) * 100,
        2
    ) AS paying_among_buyers,  -- Среднее число покупок на покупателя  Владимир, привет! исправил это значение
    ROUND(
        COUNT(e.id)::NUMERIC
        / NULLIF(COUNT(DISTINCT CASE WHEN e.amount > 0 THEN u.id END), 0),
        2
    ) AS avg_purchases_per_buyer,   -- Средняя сумма покупки
    ROUND(AVG(e.amount::NUMERIC), 2) AS avg_purchase_amount,   -- Средняя сумма на игрока (включая тех, кто не покупал)
    ROUND(SUM(e.amount)::NUMERIC / COUNT(DISTINCT u.id), 2) AS avg_spent_per_player -- Средняя сумма, потраченная на одного игрока
FROM fantasy.users u
JOIN fantasy.race r ON u.race_id = r.race_id
LEFT JOIN fantasy.events e ON u.id = e.id AND e.amount > 0
GROUP BY r.race
ORDER BY r.race;