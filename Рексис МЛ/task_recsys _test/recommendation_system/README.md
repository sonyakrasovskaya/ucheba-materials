# Recommendation System

Система персональных рекомендаций товаров на основе исторических данных о покупках пользователей.

## Описание

Проект реализует различные подходы к построению рекомендательных систем:
- Бейзлайны (Global Popular, Personal Popular)
- Collaborative Filtering (Item-based, User-based)
- Content-based фильтрация
- Гибридные подходы

## Структура проекта

- `task_recs_ds.ipynb` - основной ноутбук с EDA, реализацией алгоритмов и оценкой качества

## Метрики

Основная метрика: **Precision@3**

Дополнительные метрики:
- Recall@3
- MAP@3 (Mean Average Precision)
- NDCG@3 (Normalized Discounted Cumulative Gain)

## Результаты

Лучший алгоритм: **Personal Popular (improved)** с precision@3 = 0.3746 и ndcg@3 = 0.3968

## Особенности реализации

- Учёт частоты покупок (дубликаты как сигнал)
- Взвешенные матрицы взаимодействий
- Обработка холодного старта
- Эффективные sparse матрицы (scipy.sparse)

