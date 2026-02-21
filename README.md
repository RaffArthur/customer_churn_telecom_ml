***
[![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-FF6B35?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python_3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[![NumPy](https://img.shields.io/badge/NumPy-✓-3776AB?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-✓-3776AB?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Phik](https://img.shields.io/badge/Phi__K-✓-3776AB?logo=phik&logoColor=white)](https://phik.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-✓-3776AB?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced__learn-✓-3776AB?logo=scikit-learn&logoColor=white)](https://imbalanced-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-✓-3776AB?logo=optuna&logoColor=white)](https://optuna.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-✓-3776AB?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-3776AB?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![SHAP](https://img.shields.io/badge/SHAP-✓-3776AB?logo=shap&logoColor=white)](https://shap.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-3776AB?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![tqdm](https://img.shields.io/badge/tqdm-✓-3776AB?logo=tqdm&logoColor=white)](https://tqdm.github.io/)
[![tqdm_joblib](https://img.shields.io/badge/tqdm__joblib-✓-3776AB?logo=joblib&logoColor=white)](https://github.com/tqdm/tqdm_joblib)
***

# Разработка ML-модели для прогноза оттока клиентов телеком-оператора
Телеком-оператор борется с оттоком клиентов. Чтобы заранее находить таких пользователей была разработана модель, которая будет предсказывать, разорвёт ли абонент договор.

## Описание данных
Наборы данных:
- `contract_new.csv` — информация о договоре;
- `personal_new.csv` — персональные данные клиента;
- `internet_new.csv` — информация об интернет-услугах;
- `phone_new.csv` — информация об услугах телефонии.

**Файл `contract_new.csv`:**
- `customerID` — идентификатор абонента;
- `BeginDate` — дата начала действия договора;
- `EndDate` — дата окончания действия договора;
- `Type` — тип оплаты: раз в год-два или ежемесячно;
- `PaperlessBilling` — электронный расчётный лист;
- `PaymentMethod` — тип платежа;
- `MonthlyCharges` — расходы за месяц;
- `TotalCharges` — общие расходы абонента.

**Файл `personal_new.csv`:**
- `customerID` — идентификатор пользователя;
- `gender` — пол;
- `SeniorCitizen` — является ли абонент пенсионером;
- `Partner` — есть ли у абонента супруг или супруга;
- `Dependents` — есть ли у абонента дети.

**Файл `internet_new.csv`:**
- `customerID` — идентификатор пользователя;
- `InternetService` — тип подключения;
- `OnlineSecurity` — блокировка опасных сайтов;
- `OnlineBackup` — облачное хранилище файлов для резервного копирования данных;
- `DeviceProtection` — антивирус;
- `TechSupport` — выделенная линия технической поддержки;
- `StreamingTV` — стриминговое телевидение;
- `StreamingMovies` — каталог фильмов.

**Файл `phone_new.csv`:**
- `customerID` — идентификатор пользователя;
- `MultipleLines` — подключение телефона к нескольким линиям одновременно.

## Результаты
### Обучение ML-моделей
| Задача                       | Критерий успеха (метрика) | Тип поиска | Модель                                                                                                                                                                                                                                      | Значение метрики |
| ---------------------------- | ------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Предсказание оттока клиентов | ROC-AUC ≥ 0.90            | Optuna     | `LGBMClassifier(class_weight='balanced', learning_rate=0.11115497459571116, max_depth=48, n_estimators=256, n_jobs=-1, num_leaves=21, random_state=20226, reg_alpha=0.16465834707598878, reg_lambda=0.03689323680916021)` | **`0.9132`**     |

### Матрица ошибок и ROC-кривая
<table align="center">
  <tr>
    <td><img width="1378" height="590" alt="image" src="https://github.com/user-attachments/assets/4f27f639-3ddd-4d77-8991-3eb311506921" /></td>
  </tr>
</table>

#### Описание результатов Confussion Matrix
| Показатель         | Описание                                                          | Количество |
| ------------------ | ----------------------------------------------------------------- | ---------- |
| **True Positive**  | Модель верно определила ушедших пользователей                     | **206**    |
| **True Negative**  | Модель верно определила активных пользователей                    | **1352**   |
| **False Positive** | Модель ошиблась: назвала пользователей ушедшими, хотя они активны | **131**    |
| **False Negative** | Модель ошиблась: назвала пользователей активными, хотя они ушли   | **69**     |

#### Описание результата ROC-curve
На ROC-кривой видно, что модель обладает хорошей различительной способностью. Показатель ROC-AUC метрики подтверждает эффективность модели по сравнению со случайным угадыванием (пунктирная линий)

### SHAP & Permutation Importance анализы
#### Результаты Permutation Importance (от самого влиятельного признака до менее влиятельного)
| Признак                           | Вклад  |
| --------------------------------- | ------ |
| num__contract_duration_days       | 0.3034 |
| cat_ord__connected_services_count | 0.0175 |
| cat_ord__partner                  | 0.0140 |
| num__monthly_charges              | 0.0114 |
| cat_ord__payment_method           | 0.0092 |
| cat_ord__multiple_lines           | 0.0081 |

#### Результаты SHAP-анализа
<table align="center">
  <tr>
    <td><img width="100%" src="https://github.com/user-attachments/assets/802ee4c9-c381-44fb-80e4-a968a0059eda" /></td>
  </tr>
  <tr>
    <td><img width="100%" src="https://github.com/user-attachments/assets/78c78186-dd20-4e29-b13a-fd332f55fafe" /></td>
  </tr>
  <tr>
    <td><img width="100%" src="https://github.com/user-attachments/assets/7d3ba3bc-ac73-4dac-a6d6-bca6fc68cc53" /></td>
  </tr>
</table>

***

## Структура репозитория
```bash
customer_churn_telecom_ml/
│
├── datasets/
│ ├── contract_new.csv           # Информация о договоре
│ ├── personal_new.csv           # Персональные данные клиента
│ ├── internet_new.csv           # Информация об интернет-услугах
│ └── phone_new.csv              # Информация об услугах телефонии
├── telecom_churn_analysis.ipynb # Основной ноутбук с анализом
├── telecom_churn_analysis.py    # Конвертированный Python-скрипт
├── requirements.txt             # Зависимости проекта
├── README.md                    # Описание проекта
├── LICENSE                      # Лицензия
└── .gitignore                   # Исключения для Git
```

## Запуск
Клонирование репозитория
```bash
git clone https://github.com/RaffArthur/customer_churn_telecom_ml.git

cd customer_churn_telecom_ml
```

Установка зависимостей
```bash
pip install -r requirements.txt
```

Запуск проекта
```bash
jupyter notebook customer_churn_telecom_ml.ipynb
```
