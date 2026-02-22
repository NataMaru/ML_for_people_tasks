import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_train_val(raw_df, target_col='Exited', test_size=0.2):
    """
    Розділяє вхідний DataFrame на навчальну та валідаційну вибірки.

    Вибирає ознаки (features) з колонок з індексами [2:-1] та цільову колонку.
    Використовує стратифікацію за цільовою змінною для збереження балансу класів.

    Args:
        raw_df (pd.DataFrame): Повний набір даних.
        target_col (str): Назва цільової колонки. За замовчуванням 'Exited'.
        test_size (float): Частка даних для валідації (від 0 до 1).

    Returns:
        tuple: (X_train, y_train, X_val, y_val) — вхідні ознаки та цільові значення для обох вибірок.
    """
    input_cols = list(raw_df.columns)[2:-1]
    
    train_df, val_df = train_test_split(
        raw_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=raw_df[target_col]
    )
    
    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()
    
    return X_train, y_train, X_val, y_val

def scale_numeric_features(train_inputs, val_inputs):
    """
    Виконує масштабування числових ознак за допомогою StandardScaler.

    Навчає скейлер (fit) виключно на тренувальних даних, щоб уникнути витоку даних,
    після чого трансформує і тренувальний, і валідаційний набори.

    Args:
        train_inputs (pd.DataFrame): Навчальні вхідні ознаки.
        val_inputs (pd.DataFrame): Валідаційні вхідні ознаки.

    Returns:
        tuple: (train_inputs, val_inputs, scaler, numeric_cols)
               - Оновлені датафрейми з масштабованими числами.
               - Навчений об'єкт StandardScaler.
               - Список назв оброблених числових колонок.
    """
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    
    return train_inputs, val_inputs, scaler, numeric_cols

def encode_categorical_features(train_inputs, val_inputs):
    """
    Перетворює категоріальні ознаки у числовий формат за допомогою OneHotEncoder.

    Створює нові бінарні колонки для кожної унікальної категорії. Оригінальні 
    категоріальні колонки залишаються, а нові додаються до датафрейму.

    Args:
        train_inputs (pd.DataFrame): Навчальні вхідні ознаки.
        val_inputs (pd.DataFrame): Валідаційні вхідні ознаки.

    Returns:
        tuple: (train_inputs, val_inputs, encoder, encoded_cols)
               - Оновлені датафрейми з новими закодованими колонками.
               - Навчений об'єкт OneHotEncoder.
               - Список нових назв колонок, створених енкодером.
    """
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    
    return train_inputs, val_inputs, encoder, encoded_cols

def preprocess_data(raw_df, scaler_numeric=True):
    """
    Головна функція для повного циклу препроцесингу даних.

    Послідовно викликає функції розділення, масштабування (опціонально) та кодування. 

    Args:
        raw_df (pd.DataFrame): Початковий DataFrame з сирими даними.
        scaler_numeric (bool): Чи виконувати масштабування числових ознак. 
                               За замовчуванням True.

    Returns:
        dict: Словник з наступними ключами:
            - 'X_train' (pd.DataFrame): Фінальний набір ознак для навчання.
            - 'train_targets' (pd.Series): Цільова змінна для навчання.
            - 'X_val' (pd.DataFrame): Фінальний набір ознак для валідації.
            - 'val_targets' (pd.Series): Цільова змінна для валідації.
            - 'input_cols' (list): Список усіх назв колонок, що увійшли в фінальні X.
            - 'scaler' (StandardScaler): Об'єкт, використаний для масштабування.
            - 'encoder' (OneHotEncoder): Об'єкт, використаний для кодування.

              Якщо scaler_numeric=False, ключ 'scaler' буде містити None.    

    Example:
        >>> data = preprocess_data(df)
        >>> model.fit(data['X_train'], data['train_targets'])
    """


    
    # 1. Розбиття
    X_train_raw, y_train, X_val_raw, y_val = split_train_val(raw_df)
    
    # 2. Масштабування (виконується лише якщо scaler_numeric=True)
    if scaler_numeric:
        X_train_scaled, X_val_scaled, scaler, num_cols = scale_numeric_features(X_train_raw, X_val_raw)
    else:
        X_train_scaled, X_val_scaled = X_train_raw, X_val_raw
        scaler = None
        num_cols = X_train_raw.select_dtypes(include=np.number).columns.tolist()
    
    # 3. Кодування
    X_train_final, X_val_final, encoder, cat_cols = encode_categorical_features(X_train_scaled, X_val_scaled)
    
    # 4. Об'єднання списку колонок
    input_cols = num_cols + cat_cols
    
    return {
        'X_train': X_train_final[input_cols],
        'train_targets': y_train,
        'X_val': X_val_final[input_cols],
        'val_targets': y_val,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(new_df, input_cols, scaler, encoder):
    """
    Обробляє нові дані за допомогою вже навчених скейлера та енкодера.

    Ця функція використовує лише метод .transform(), що забезпечує ідентичність 
    обробки нових даних відносно тренувальних.

    Args:
        new_df (pd.DataFrame): Нові сирі дані (наприклад, з test.csv).
        input_cols (list): Фінальний список колонок (numeric + encoded), 
                           який очікує модель.
        scaler (StandardScaler): Навчений скейлер з результатів preprocess_data.
        encoder (OneHotEncoder): Навчений енкодер з результатів preprocess_data.

    Returns:
        pd.DataFrame: Оброблений набір даних, готовий до подачі в модель (X_new).
    """
    # 1. Відбираємо ознаки 
    raw_input_cols = list(new_df.columns)[2:]
    X_new = new_df[raw_input_cols].copy()

    # 2. Визначаємо типи колонок
    numeric_cols = X_new.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_new.select_dtypes(include=['object']).columns.tolist()

    # 3. Масштабування числових ознак
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    # 4. Кодування категоріальних ознак 
    encoded_features = encoder.transform(X_new[categorical_cols])
    encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))
    
    # Створюємо DataFrame з закодованих ознак та об'єднуємо
    X_encoded_df = pd.DataFrame(encoded_features, columns=encoded_col_names, index=X_new.index)
    
    # Формуємо фінальний DataFrame
    X_final = pd.concat([X_new[numeric_cols], X_encoded_df], axis=1)

    # 5. Гарантуємо порядок колонок та наявність лише потрібних фіч
    return X_final[input_cols]
