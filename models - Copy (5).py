# models.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
import kerastuner as kt

def build_lstm_model(hp, look_back, n_features):
    """สร้าง LSTM model พร้อมปรับจูน hyperparameters ด้วย KerasTuner"""
    model = Sequential()
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
                   return_sequences=True,
                   input_shape=(look_back, n_features)))
    model.add(Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('lstm_units2', min_value=32, max_value=128, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_rate2', 0.1, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='LOG')),
                  loss='mean_squared_error')
    return model

def build_cnn_lstm_model(look_back, n_features):
    """สร้างโมเดล CNN-LSTM"""
    inputs = Input(shape=(look_back, n_features))
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_transformer_model(look_back, n_features):
    """สร้างโมเดล Transformer แบบง่าย"""
    inputs = Input(shape=(look_back, n_features))
    # ใช้ MultiHeadAttention โดยใช้ inputs เป็น key, query, value
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_ensemble_model(models_list, look_back, n_features):
    """รวมผลทำนายจากโมเดลหลายตัวเป็น ensemble"""
    inputs = Input(shape=(look_back, n_features))
    predictions = [model(inputs) for model in models_list]
    averaged = tf.keras.layers.Average()(predictions)
    ensemble_model = Model(inputs=inputs, outputs=averaged)
    ensemble_model.compile(optimizer='adam', loss='mean_squared_error')
    return ensemble_model
