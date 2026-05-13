import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, accuracy_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import os

SEED = 67
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ============================================================
# 1. ЗАГРУЗКА И ОБРАБОТКА ДАТАСЕТА
# ============================================================
print("=" * 60)
print("1. ЗАГРУЗКА И ОБРАБОТКА ДАТАСЕТА")
print("=" * 60)

df = pd.read_csv("datasets/ai_vs_human_text_2026.csv")
print(f"\nРазмер: {df.shape}")
print(f"\Признаков: {list(df.columns)}")
print(f"\nМетки:\n{df['label'].value_counts()}")
print(f"\nNull значения:\n{df.isnull().sum()}")
print(f"\nПримеры текста:")
for i in range(3):
    print(f"  [{df['label'].iloc[i]}] {df['text_content'].iloc[i][:80]}...")

# ============================================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================================
print("\n" + "=" * 60)
print("2. ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

# Encode labels: human -> 0, ai -> 1
df["label_encoded"] = (df["label"] == "ai").astype(int)

# Split: 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df["text_content"], df["label_encoded"],
    test_size=0.15, random_state=42, stratify=df["label_encoded"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1765, random_state=42, stratify=y_train_val  # 0.1765 * 0.85 = 0.15
)

print(f"Примеров для обучения: {len(X_train)}")
print(f"Примеров для валидации: {len(X_val)}")
print(f"Примеров для теста: {len(X_test)}")

# Tokenize
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print(f"Размер словаря: {len(tokenizer.word_index)}")
print(f"Максимальная длина последовательности: {MAX_LEN}")

# ============================================================
# 3. СОЗДАНИЕ LSTM МОДЕЛИ
# ============================================================
print("\n" + "=" * 60)
print("3. СОЗДАНИЕ LSTM МОДЕЛИ")
print("=" * 60)

EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.3

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(DROPOUT_RATE),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
    Dropout(DROPOUT_RATE),
    Dense(32, activation="relu"),
    Dropout(DROPOUT_RATE / 2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 4. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
print("\n" + "=" * 60)
print("4. ОБУЧЕНИЕ МОДЕЛИ")
print("=" * 60)

EPOCHS = 10
BATCH_SIZE = 32

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 5. ПРОВЕРКА НА ТЕСТОВЫХ ДАННЫХ
# ============================================================
print("\n" + "=" * 60)
print("5. ПРОВЕРКА НА ТЕСТОВЫХ ДАННЫХ")
print("=" * 60)

y_pred_prob = model.predict(X_test_pad).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = y_test.values

accuracy = accuracy_score(y_true, y_pred)
print(f"\nТочность на тестовых данных: {accuracy:.4f}")
print(f"\nОтчёт классификации:")
print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
os.environ["TCL_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

print("\n" + "=" * 60)
print("6. ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LSTM Model — AI vs Human Text Classification", fontsize=16, fontweight="bold")

# --- Plot 1: Training & Validation Accuracy ---
ax1 = axes[0, 0]
ax1.plot(history.history["accuracy"], label="Точность (тестовая)", linewidth=2, color="#2196F3")
ax1.plot(history.history["val_accuracy"], label="Точность (валидационная)", linewidth=2, color="#FF5722")
ax1.set_title("Точность по эпохам", fontsize=13)
ax1.set_xlabel("Эпоха")
ax1.set_ylabel("Точность")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Training & Validation Loss ---
ax2 = axes[0, 1]
ax2.plot(history.history["loss"], label="Train Loss", linewidth=2, color="#2196F3")
ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2, color="#FF5722")
ax1.set_title("Потери по эпохам", fontsize=13)
ax1.set_xlabel("Эпоха")
ax1.set_ylabel("Потери")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Confusion Matrix ---
# ax3 = axes[1, 0]
# cm = confusion_matrix(y_true, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
#             xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
# ax3.set_title("Confusion Matrix", fontsize=13)
# ax3.set_ylabel("True Label")
# ax3.set_xlabel("Predicted Label")

# --- Plot 4: ROC Curve ---
ax4 = axes[1, 1]
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, linewidth=2, color="#4CAF50", label=f"ROC (AUC = {roc_auc:.4f})")
ax4.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
ax4.set_title("ROC Кривая", fontsize=13)
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend(loc="lower right")
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("lstm_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nГрафик сохранён как 'lstm_results.png'")

# ============================================================
# 7. ТЕСТЫ НА ПРИМЕРАХ
# ============================================================
print("\n" + "=" * 60)
print("7. ТЕСТЫ НА ПРИМЕРАХ")
print("=" * 60)

demo_texts = [
    "Can we talk about gene editing ethics for a sec because i feel like people are completely missing the point here.",
    "Analysts are closely watching developments related to climate change adaptation strategies, noting that recent events mark a potential turning point.",
    "I just grabbed coffee and walked my dog around the block, what a beautiful morning!",
    "This study investigates the multifaceted dimensions employing a mixed-methods approach to capture both quantitative patterns."
]

demo_seq = tokenizer.texts_to_sequences(demo_texts)
demo_pad = pad_sequences(demo_seq, maxlen=MAX_LEN, padding="post", truncating="post")
demo_probs = model.predict(demo_pad).ravel()

print(f"\n  AI == 1 | Human == 0\n")

for text, prob in zip(demo_texts, demo_probs):
    label = "AI" if prob >= 0.5 else "Human"
    print(f"\n  Текст: {text}")
    print(f"  -> Метка: {label} (Сырая метка: {prob:.4f})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)