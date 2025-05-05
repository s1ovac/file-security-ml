import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Настройка для воспроизводимости
np.random.seed(42)

class FileSecurityAnalyzer:
   def __init__(self):
       self.model = None
       self.scaler = StandardScaler()
       self.label_encoders = {}
       
   def generate_synthetic_data(self, n_samples=10000):
       """Генерация синтетических данных о файлах"""
       print("Генерация синтетических данных...")
       
       # Определение типов файлов и их характеристик
       safe_extensions = ['pdf', 'docx', 'xlsx', 'txt', 'png', 'jpg', 'zip']
       unsafe_extensions = ['exe', 'bat', 'vbs', 'js', 'scr', 'ps1', 'cmd']
       
       # Генерация данных
       data = []
       for _ in range(n_samples):
           # Выбор безопасности
           is_safe = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% безопасных файлов
           
           # Выбор расширения
           if is_safe:
               extension = np.random.choice(safe_extensions)
           else:
               extension = np.random.choice(unsafe_extensions)
           
           # Размер файла (в КБ)
           if is_safe:
               file_size = np.random.exponential(500) + 10  # Среднее 500 КБ
           else:
               file_size = np.random.exponential(200) + 5000  # Больше для вредоносных
           
           # Возраст файла (дни)
           file_age = np.random.exponential(30)
           
           # Количество обращений к файлу
           access_count = np.random.poisson(5) if is_safe else np.random.poisson(20)
           
           # Уровень сжатия (для архивов)
           compression_ratio = np.random.uniform(0.1, 0.9) if extension == 'zip' else 0
           
           # Имеет ли цифровую подпись
           has_signature = 1 if (is_safe and np.random.random() > 0.3) else 0
           
           # Добавление случайного шума для более реалистичных данных
           if np.random.random() < 0.1:  # 10% шума
               is_safe = 1 - is_safe
           
           data.append({
               'extension': extension,
               'file_size': file_size,
               'file_age': file_age,
               'access_count': access_count,
               'compression_ratio': compression_ratio,
               'has_signature': has_signature,
               'is_safe': is_safe
           })
       
       return pd.DataFrame(data)
   
   def perform_eda(self, df):
       """Анализ данных"""
       print("\n=== Анализ данных ===")
       
       # Общая статистика
       print(f"Всего файлов: {len(df)}")
       print(f"Безопасных файлов: {df['is_safe'].sum()} ({df['is_safe'].mean()*100:.1f}%)")
       print(f"Опасных файлов: {len(df) - df['is_safe'].sum()} ({(1-df['is_safe'].mean())*100:.1f}%)")
       
       # Визуализация
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
       # Распределение по расширениям
       ext_counts = df['extension'].value_counts()
       ext_safety = df.groupby('extension')['is_safe'].mean()
       
       axes[0,0].bar(ext_counts.index, ext_counts.values)
       axes[0,0].set_title('Распределение по расширениям файлов')
       axes[0,0].set_xticklabels(ext_counts.index, rotation=45)
       
       # Безопасность по расширениям
       axes[0,1].bar(ext_safety.index, ext_safety.values)
       axes[0,1].set_title('Процент безопасных файлов по расширениям')
       axes[0,1].set_xticklabels(ext_safety.index, rotation=45)
       
       # Распределение размеров файлов
       axes[1,0].hist(df[df['is_safe']==1]['file_size'], bins=50, alpha=0.5, label='Безопасные')
       axes[1,0].hist(df[df['is_safe']==0]['file_size'], bins=50, alpha=0.5, label='Опасные')
       axes[1,0].set_title('Распределение размеров файлов')
       axes[1,0].set_xlabel('Размер (КБ)')
       axes[1,0].legend()
       
       # Корреляционная матрица
       numeric_cols = ['file_size', 'file_age', 'access_count', 'compression_ratio', 'has_signature', 'is_safe']
       corr_matrix = df[numeric_cols].corr()
       sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,1])
       axes[1,1].set_title('Корреляционная матрица')
       
       plt.tight_layout()
       plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
       plt.close()
       
       return df
   
   def preprocess_data(self, df):
       """Предобработка данных"""
       print("\n=== Предобработка данных ===")
       
       # Кодирование категориальных признаков
       df_processed = df.copy()
       
       for column in ['extension']:
           le = LabelEncoder()
           df_processed[column] = le.fit_transform(df_processed[column])
           self.label_encoders[column] = le
       
       # Разделение на признаки и целевую переменную
       X = df_processed.drop('is_safe', axis=1)
       y = df_processed['is_safe']
       
       # Нормализация числовых признаков
       numeric_features = ['file_size', 'file_age', 'access_count', 'compression_ratio']
       X_scaled = X.copy()
       X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
       
       return X_scaled, y
   
   def train_model(self, X, y):
       """Обучение модели"""
       print("\n=== Обучение модели ===")
       
       # Разделение данных
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
       # Создание модели Random Forest
       self.model = RandomForestClassifier(n_estimators=100, random_state=42)
       
       # Обучение модели
       print("\nОбучение Random Forest...")
       self.model.fit(X_train, y_train)
       
       # Проверка точности
       score = self.model.score(X_test, y_test)
       print(f"\nТочность Random Forest: {score:.4f}")
       
       return X_test, y_test
   
   def evaluate_model(self, X_test, y_test):
       """Оценка модели"""
       print("\n=== Оценка модели ===")
       
       # Предсказания
       y_pred = self.model.predict(X_test)
       y_pred_proba = self.model.predict_proba(X_test)[:, 1]
       
       # Метрики
       print("\nМетрики классификации:")
       print(classification_report(y_test, y_pred))
       
       # ROC AUC
       roc_auc = roc_auc_score(y_test, y_pred_proba)
       print(f"\nROC AUC Score: {roc_auc:.4f}")
       
       # Визуализация результатов
       fig, axes = plt.subplots(1, 2, figsize=(15, 6))
       
       # Матрица ошибок
       cm = confusion_matrix(y_test, y_pred)
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
       axes[0].set_title('Матрица ошибок')
       axes[0].set_xlabel('Предсказаные значения')
       axes[0].set_ylabel('Истинные значения')
       
       # ROC кривая
       fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
       axes[1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
       axes[1].plot([0, 1], [0, 1], 'k--')
       axes[1].set_title('ROC кривая')
       axes[1].set_xlabel('False Positive Rate')
       axes[1].set_ylabel('True Positive Rate')
       axes[1].legend()
       
       plt.tight_layout()
       plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
       plt.close()
       
       # Важность признаков
       feature_importance = pd.DataFrame({
           'feature': X_test.columns,
           'importance': self.model.feature_importances_
       })
       feature_importance = feature_importance.sort_values('importance', ascending=False)
       
       plt.figure(figsize=(10, 6))
       sns.barplot(data=feature_importance, x='importance', y='feature')
       plt.title('Важность признаков')
       plt.tight_layout()
       plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
       plt.close()
       
       return y_pred_proba
   
   def predict_file_safety(self, file_data):
       """Предсказание безопасности файла"""
       if self.model is None:
           raise ValueError("Модель не обучена!")
       
       # Предобработка данных
       file_df = pd.DataFrame([file_data])
       
       # Кодирование категориальных признаков
       for column in ['extension']:
           if column in self.label_encoders:
               file_df[column] = self.label_encoders[column].transform(file_df[column])
       
       # Нормализация
       numeric_features = ['file_size', 'file_age', 'access_count', 'compression_ratio']
       file_df[numeric_features] = self.scaler.transform(file_df[numeric_features])
       
       # Предсказание
       safety_probability = self.model.predict_proba(file_df)[0][1]
       
       return safety_probability

# Основной код выполнения
if __name__ == "__main__":
   print("Система анализа безопасности файлов в облачном хранилище")
   print("=" * 50)
   
   # Создание анализатора
   analyzer = FileSecurityAnalyzer()
   
   # Генерация данных
   data = analyzer.generate_synthetic_data(n_samples=10000)
   
   # Анализ данных
   data = analyzer.perform_eda(data)
   
   # Предобработка
   X, y = analyzer.preprocess_data(data)
   
   # Обучение модели
   X_test, y_test = analyzer.train_model(X, y)
   
   # Оценка модели
   y_pred_proba = analyzer.evaluate_model(X_test, y_test)
   
   # Пример предсказания
   print("\n=== Пример предсказания ===")
   test_files = [
       {
           'extension': 'exe',
           'file_size': 5000,
           'file_age': 1,
           'access_count': 100,
           'compression_ratio': 0,
           'has_signature': 0
       },
       {
           'extension': 'pdf',
           'file_size': 100,
           'file_age': 30,
           'access_count': 5,
           'compression_ratio': 0,
           'has_signature': 1
       }
   ]
   
   for i, test_file in enumerate(test_files):
       safety_prob = analyzer.predict_file_safety(test_file)
       print(f"\nФайл {i+1}: {test_file}")
       print(f"Вероятность безопасности файла: {safety_prob:.4f}")
       print(f"Классификация: {'Безопасный' if safety_prob > 0.5 else 'Опасный'}")
   
   print("\nРезультаты сохранены в файлы:")
   print("- eda_visualization.png")
   print("- model_evaluation.png")
   print("- feature_importance.png")