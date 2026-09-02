# -*- coding: utf-8 -*-
"""Keras 快速上手 —— 演示代码 (TensorFlow 的 tf.keras)

覆盖要点：
1. Sequential 与 Functional 两种建模型方式;
2. compile(optimizer/loss/metrics) 与 fit(epochs/batch_size/validation) 的约定;
3. 回调三件套: EarlyStopping / ReduceLROnPlateau / ModelCheckpoint;
4. evaluate / predict 的使用与分类 argmax;
5. 整数标签要用 sparse_categorical_crossentropy(不必 one-hot);
6. 模型保存与加载。

数据: 本脚本用合成随机数据保证离线可跑; 若装有网络, 注释里给出换成 MNIST 的写法。

运行:
    python Keras快速上手_sample.py
"""
import os

import numpy as np
import tensorflow as tf

# 关掉 INFO 级噪音(可选)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def make_synthetic_data(n=3000, n_classes=10):
    """合成"伪MNIST": 28x28 特征 + 整数标签 0..9(标签噪声较高, 只为演示 API)。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 28, 28))
    y = rng.integers(0, n_classes, size=(n,))
    return x, y


def demo_sequential():
    print("=" * 66)
    print("实验1: Sequential —— 层叠直筒模型")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),   # 28x28 -> 784 (不含batch维)
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"), # 多分类概率输出
    ])
    model.summary(line_length=90)
    return model


def demo_functional():
    print("=" * 66)
    print("实验2: Functional —— 图式模型(多输入演示: 展平+全局统计量拼接)")
    inp = tf.keras.Input(shape=(28, 28), name="pixels")
    flat = tf.keras.layers.Flatten()(inp)
    stats = tf.keras.layers.Lambda(
        lambda t: tf.stack([tf.reduce_mean(t, axis=(1, 2)),
                            tf.reduce_std(t, axis=(1, 2))], axis=1))(inp)
    x = tf.keras.layers.concatenate([flat, stats])       # 784 + 2
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(10, activation="softmax", name="logits")(x)
    model = tf.keras.Model(inp, out)
    model.summary(line_length=90)
    return model


def demo_train_eval(model, x, y):
    print("=" * 66)
    print("实验3: compile + fit(含回调) + evaluate + predict")

    # 整数标签 + sparse 交叉熵; 若用 one-hot 才换 categorical_crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 回调: 早停(看val, 恢复最佳权重) / 指标停滞降lr / 存最优模型
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("keras_demo_best.keras",
                                           monitor="val_accuracy",
                                           save_best_only=True),
    ]

    split = int(len(x) * 0.8)
    history = model.fit(
        x[:split], y[:split],
        batch_size=64,
        epochs=10,
        validation_data=(x[split:], y[split:]),   # 显式验证集
        callbacks=callbacks,
        verbose=0,
    )
    print(f"    训练完成: 末轮 train_acc={history.history['accuracy'][-1]:.3f}, "
          f"val_acc={history.history['val_accuracy'][-1]:.3f}")
    print(f"    实际跑的epoch数={len(history.history['loss'])} (EarlyStopping可能提前停)")

    loss, acc = model.evaluate(x[split:], y[split:], verbose=0)
    print(f"    evaluate: loss={loss:.3f} acc={acc:.3f}")

    prob = model.predict(x[:3], verbose=0)
    pred = prob.argmax(axis=1)
    print(f"    predict: 前3条概率形状{prob.shape} -> argmax 预测 {pred.tolist()}, "
          f"真值 {y[:3].tolist()}")
    return history


def demo_save_load(model, x, y):
    print("=" * 66)
    print("实验4: 保存与加载 (ModelCheckpoint 产物)")
    # model.save 保存整模型(架构+权重); load_model 可直接恢复
    model.save("keras_demo_final.keras")
    m2 = tf.keras.models.load_model("keras_demo_final.keras")
    p1 = model.predict(x[:2], verbose=0)
    p2 = m2.predict(x[:2], verbose=0)
    print(f"    加载后预测与保存前一致: "
          f"{np.allclose(p1, p2, atol=1e-6)}")


def swap_for_real_mnist_comment():
    """提示: 换真实 MNIST 只需两行(需联网下载数据集)。"""
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.astype("float32") / 255.0   # 归一化到 [0,1]
    # 再把上面 fit 的 x/y 换成 (x_train, y_train), validation_data=(x_test, y_test)
    print("=" * 66)
    print("实验5: 换真实数据提示")
    print("    合成数据标签与特征无关, acc 约等于随机 -> 本脚本只为演示 API 流程;")
    print("    跑真实 MNIST: 见 swap_for_real_mnist_comment() 注释里的两行替换。")


if __name__ == "__main__":
    print(f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}\n")
    x_data, y_data = make_synthetic_data()
    seq_model = demo_sequential()
    # 用 Functional 模型继续演示(Sequential 同样适用下面所有 API)
    fn_model = demo_functional()
    demo_train_eval(fn_model, x_data, y_data)
    demo_save_load(fn_model, x_data, y_data)
    swap_for_real_mnist_comment()
    # 清理演示产物
    for f in ("keras_demo_best.keras", "keras_demo_final.keras"):
        if os.path.exists(f):
            os.remove(f)
    print("    (已清理 keras_demo_*.keras 演示文件)")
