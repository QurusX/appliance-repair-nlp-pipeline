# Appliance Repair NLP Pipeline 🛠️🤖

**Автоматизированная система полного цикла для классификации, категоризации и извлечения технических сущностей из данных по ремонту бытовой техники.**

Этот проект решает задачу обработки огромных массивов неструктурированных («грязных») данных с форумов и сервисов вопросов-ответов. Система превращает сырой текст в структурированную базу данных, готовую для использования в поисковых движках или базах знаний.

## 📖 Обзор подхода

В основе проекта лежит концепция **дистилляции знаний (Knowledge Distillation)**:
1.  **Учитель:** API Google Gemini 2.5 Flash размечает небольшую выборку данных с высочайшей точностью.
2.  **Ученик:** Локальные легковесные модели (**DistilBERT, T5**) обучаются на этих данных, чтобы выполнять задачу полностью офлайн, бесплатно и с очень высокой скоростью.

## ✨ Ключевые возможности

*   **Умная фильтрация:** Автоматическое определение, относится ли запрос к теме ремонта (бинарная классификация).
*   **Генеративная категоризация:** Вместо жесткого списка категорий модель T5 генерирует чистые, профессиональные заголовки проблем (например, *«Стук компрессора холодильника»*).
*   **Извлечение технических сущностей (NER):** Гибридный подход (Transformer NER + Regex) для поиска:
    *   **Номеров моделей** (например, `WFW6620HW0`)
    *   **Артикулов запчастей** (например, `WPW10448876`)
    *   **Кодов ошибок** (например, `F8 E1`, `OE`)
*   **Анализ решений:** Определение наличия конкретных шагов по устранению неисправности в тексте.
*   **Высокая производительность:** Обработка **13,000+ записей менее чем за 4 минуты** на одной видеокарте (RTX 3060) благодаря пакетной обработке (Batching).

## 📊 Примеры обработки данных (Pipeline Examples)

Система эффективно обрабатывает связки «Вопрос-Ответ», извлекая суть даже из неструктурированных и длинных текстов.

| Вопрос пользователя (Question) | Ответ мастера (Answer) | Категория (AI Category) | Извлеченные Part Numbers |
| :--- | :--- | :--- | :--- |
| "I need to buy an upper rack?" | "Hi Tom... The Upper Rack is listed as PartSelect #: **PS12348079**. Good luck with the repair!" | **Part Identification** | `PS12348079` |
| "Can’t get the drum off to put the belt on what else can I do?" | "Hello Benita... To remove the drum, you will use a putty knife and push the spring clips... Disconnect the belt then lift on the drum..." | **belt replacement** | - |
| "Washer bang and knocks while in spin cycle" | "Hi Michael... It appears that your washer may have a damaged suspension spring, part number **PS11751118**..." | **Spin cycle bang and knocks** | `PS11751118` |
| "2 questions about fabric softener dispenser. It does not drain all the way..." | "Hi Laurie... For a deep clean, soak the dispenser parts in a solution of warm, soapy water for 15–30 minutes..." | **dispenser not draining** | - |
| "My GE refrigerator model GNE27JYMFS. Can someone tell me the width?" | "The GE GNE27JYMFS refrigerator is approximately 35.75 inches wide... check the product page for more info." | **Non-repair context** | - |

### Что здесь происходит:
1.  **Смысловой анализ:** В примере с барабаном (drum) ИИ проанализировал длинную инструкцию и понял, что речь идет о **замене ремня (belt replacement)**.
2.  **Точное извлечение:** Из примера с верхней корзиной (upper rack) модель мгновенно выделила технический артикул `PS12348079`.
3.  **Фильтрация шума:** Вопрос про габариты холодильника был автоматически классифицирован как **не связанный с ремонтом**, что позволяет очистить базу данных от нецелевого контента.

## 🛠 Технологический стек

*   **Ядро:** Python 3.10, PyTorch
*   **NLP:** Hugging Face Transformers, Datasets, Tokenizers
*   **Модели:** DistilBERT (Классификация и NER), T5-Small (Генерация текста)
*   **API:** Google GenAI SDK (Gemini 2.5 Flash Lite)
*   **Данные:** Pandas, Openpyxl, Scikit-learn

## 🚀 Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/ваш_логин/appliance-repair-nlp-pipeline.git
    cd appliance-repair-nlp-pipeline
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Настройте переменные окружения:**
    Создайте файл `.env` в корневой папке:
    ```text
    GEMINI_API_KEY=ваш_ключ_api
    ```

## 💻 Использование

Управление проектом осуществляется через единую точку входа `main.py` в трех режимах:

### 1. Разметка данных (Режим API)
Создание высококачественного обучающего набора данных с помощью Gemini API.
```bash
python main.py --mode labeling
```

### 2. Обучение моделей (Режим Local ML)
Обучение локальных моделей-«учеников» на вашей видеокарте.
```bash
python main.py --mode train
```

### 3. Промышленный запуск (Режим Inference)
Масштабная обработка новых Excel-файлов без затрат на API и интернет.
```bash
python main.py --mode run --input data/raw/new_data.xlsx --output data/processed/results.xlsx
```

## 📁 Структура проекта

```text
├── data/
│   ├── raw/            # Исходные Excel файлы
│   └── processed/      # Размеченные данные и финальные результаты
├── models/             # Веса обученных локальных моделей
├── src/
│   ├── labeler.py      # Интеграция с Gemini API
│   ├── trainer.py      # Логика обучения BERT/T5
│   ├── extractor.py    # Извлечение сущностей (NER + Regex)
│   └── pipeline.py     # Оркестрация пакетной обработки
├── config.py           # Глобальные настройки и паттерны
├── main.py             # Точка входа CLI
└── requirements.txt    # Зависимости проекта
```

## 📊 Показатели производительности

| Задача | Модель | Скорость (RTX 3060) | Точность (отн. Gemini) |
| :--- | :--- | :--- | :--- |
| Классификация | DistilBERT | ~3500 зап/мин | 96% |
| Категоризация | T5-Small | ~1200 зап/мин | 91% |
| Извлечение данных| Hybrid (NER+Reg) | ~4000 зап/мин | 94% |

---

### 🤝 Контакты
Разработано: [Николай Алымов] — [t.me/nickalymov]
Буду рад сотрудничеству в области автоматизации данных и внедрения ИИ-решений!