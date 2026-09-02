---
title: "Keras 快速上手：Sequential / Functional、compile / fit、回调"
tags: [TensorFlow, Keras, 快速上手]
date: 2026-08-29
---

# Keras 快速上手：Sequential / Functional、compile / fit、回调

## 定义

- **是什么**：Keras 是构建在 TensorFlow 之上的高层神经网络 API（现通常以 `tf.keras` 形态出现；Keras 3 亦可作为独立库选择 TensorFlow / JAX / PyTorch 后端）。它是一套"配置式"高层接口：**搭模型像搭积木，训练像填表单**——`Sequential` 顺序堆层即可用，遇到多输入/多输出/共享层再升级 `Functional`。
- **解决什么问题**：手写训练需要把前向传播、反向传播、批次切分、指标累计、学习率调度等易错且高度重复的细节全部暴露给用户。Keras 把"神经网络训练"抽象成三个固定动作：**建模型 → compile（声明优化器/损失/指标）→ fit（喂数据）**，把易错的循环、反向传播、批次逻辑全部封装掉。
- **核心特征**：① 配置式/声明式——用"配置"代替手写训练循环，约定优于代码；② 大量隐藏默认约定降低门槛——输入按 `(batch, ...)` 隐式带 batch 维、默认 channels_last、`metrics` 传字符串自动映射；③ 分层抽象——上层的产物（模型、层）在自定义 Layer/Model、`tf.GradientTape` 自定义训练循环等下层仍能复用。
- **适用范畴**：标准监督学习任务（图像/表格分类、回归、多任务多输出），快速原型与五分钟跑通 MNIST 级任务；`Sequential` 只适合"一条主线走到底"，`Functional` 支持多输入、多输出、分支合并（concat/add）、共享层、残差结构。
- **边界**：当需要细粒度控制（自定义训练循环、梯度裁剪、特殊更新规则）时才下钻到 `tf.GradientTape`；此时 compile/fit 的产物（层、优化器）依然可复用，两者不是互斥而是分层。

## 原理

- **为什么这样设计**：训练神经网络的公共骨架——按 batch 迭代、前向 → 损失 → 反向 → 参数更新、逐 epoch 记录指标——是高度重复且最容易写错的代码。Keras 把它收敛进一个 `fit` 循环，把真正的差异点（网络结构、损失、优化器、指标、调度策略）变成"可配置参数"。因此 `compile` 只是**声明**这些差异（`loss` / `optimizer` / `metrics` 支持字符串、实例或自定义函数），不执行任何训练；真正的批次循环与反向传播发生在 `fit` 被调用之后。
- **Sequential 机制**：内部按添加顺序把层串成一条线性管道，逐层传递张量；第一层必须能推断输入形状（`input_shape` 不含 batch 维——真实数据 `(batch, 28, 28)` 时只声明 `(28, 28)`）。
- **Functional 机制**：层是可调用对象，`Flatten()(inp)` 这类"张量接线"会逐步构造一张计算图（图中的 KerasTensor 携带形状元信息），`Input` 是图入口，`Model(inputs, outputs)` 把子图固化为可训练模型。因为是显式图结构，所以天然支持 DAG：`concat`/`add` 分支合并、多输入多输出、同一层实例被多处引用（共享层）、残差连接。**任何 Sequential 能做的它都能做**，故用 Functional 是安全默认。
- **损失与标签编码的配对原理**：`compile` 里的字符串（`"adam"`、`"sparse_categorical_crossentropy"`、`"accuracy"`）由框架符号表映射到具体实现；交叉熵需要把标签编码与损失版本对齐——整数标签 0..9 用 **sparse** 版，one-hot 向量用 **categorical** 版。softmax + crossentropy 的数值稳定性（如 logits 直接算交叉熵避免中间 softmax 下溢）由框架 fused 实现处理，不要在自定义时重复实现。
- **回调（Callback）机制**：callback 对象挂在 `fit` 循环的各阶段钩子上（epoch/batch 开始与结束等）。`fit` 结束返回的是**当下那一刻**的权重，所以早停必须配合 `restore_best_weights=True`，否则拿到的是触发早停时的权重而非监控指标最优的那一版；`monitor` 应指向验证集（`val_loss`）而不是训练集。
- **保存格式**：Keras 3 使用新格式 `model.keras`（单文件、自包含、含结构与权重），同时兼容老式 `"xxx.h5"`；`load_model` 后可继续 evaluate / predict / fit。

## 应用

**典型场景**：图像/表格分类、回归、多输入多输出建模，以及需要"先快速跑通、再决定是否降级手写循环"的一切标准监督学习任务。**快速上手五步**：① 用 Sequential（简单直筒）或 Functional（图式，安全默认）建模型，第一层给 `input_shape`；② `compile(optimizer="adam", loss=..., metrics=[...])`；③ `fit(epochs, batch_size, validation_split/validation_data, callbacks)` 拿到返回的 `History`（含逐 epoch loss/acc）；④ `evaluate` 出 loss+metric，`predict` 出概率/值，分类任务取 `argmax(axis=1)`；⑤ `model.save("model.keras")` 与 `load_model` 完成复用。核心三件套参数对照：

| 步骤 | 关键参数 | 说明 |
| --- | --- | --- |
| `compile` | `optimizer="adam"`、`loss=...`、`metrics=["accuracy"]` | 字符串即可，如 `loss="sparse_categorical_crossentropy"`；标签是整数时用 sparse 版（one-hot 用 categorical） |
| `fit` | `epochs`、`batch_size`、`validation_split`/`validation_data`、`callbacks`、`shuffle=True` | 返回 `History`（含逐 epoch loss/acc）；`validation_data` 比 `validation_split` 更可控 |
| `evaluate` / `predict` | `model.evaluate(x_test, y_test)`、`model.predict(x)` | evaluate 出 loss+metric；predict 出概率/值，分类取 `argmax(axis=1)` |

训练中常用回调把"断点/早停/降 lr/日志"从训练循环里解耦出去：

| 回调 | 作用 | 典型用法 |
| --- | --- | --- |
| `EarlyStopping` | 监控指标不改善则提前停 | `monitor="val_loss", patience=5, restore_best_weights=True` |
| `ModelCheckpoint` | 每轮存权重/整模型 | `monitor="val_acc", save_best_only=True` |
| `ReduceLROnPlateau` | 指标停滞降 lr | `factor=0.5, patience=3` |
| `TensorBoard` | 训练可视化日志 | `log_dir="logs"` |
| `LearningRateScheduler` | 自定义 lr 策略 | 传入 `schedule(epoch, lr)` 函数 |

**常见坑 ❌✅**：

- ❌ 标签是整数 0..9 却用 `categorical_crossentropy` 且不 one-hot → 报错/指标异常。
- ✅ 整数标签 → `sparse_categorical_crossentropy`；one-hot → `categorical_crossentropy`。
- ❌ 用 `Input(shape=(28, 28))` 后再写 `Flatten(input_shape=...)` 双重指定 / 或漏掉 batch 维。
- ✅ 惯例：`input_shape`/`Input.shape` **不含 batch 维**；第一层给出即可。
- ❌ 回归任务拿 `accuracy` 当指标、或输出层忘了去掉 softmax。
- ✅ 回归：`loss="mse"`、输出层无激活、指标 `mae` 等。
- ❌ 只在训练集上 EarlyStopping 的 monitor。
- ✅ 用验证集：`monitor="val_loss"`，并 `restore_best_weights=True`（否则返回的是早停那一刻的权重，不一定是最好的一版）。
- ❌ 混淆 `model.predict`（推理，自动按 batch 循环）与 `model(x)`（Keras 3 里直接张量调用）。
- ❌ 每次 fit 前忘 `model.compile` 或改 loss 忘了重新 compile。
- ✅ compile 是"声明一次、随时可重来"：改学习率/损失先再 compile 一次。
- ❌ 内存中一次性 `np.array` 全量数据 → 大数集 OOM。
- ✅ 用 `tf.data.Dataset`（`from_tensor_slices` + `batch/prefetch/shuffle`）喂 fit。
- ✅ 训练前 `tf.config.set_visible_devices`/显存增长设置，避免多卡抢显存（或直接让 TF 自动）。

```python
# ==================== ① Sequential 顺序式建模型（层叠直筒） ====================
# 只适合"一条主线走到底"；第一层必须能推断输入形状。
# input_shape 不含 batch 维：真实数据 shape=(batch, 28, 28)，这里只写 (28, 28)。
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # 首层：把 28×28 拉平成 784 向量
    tf.keras.layers.Dense(128, activation="relu"),   # 隐层：128 神经元
    tf.keras.layers.Dropout(0.2),                    # 随机丢弃 20% 神经元，防过拟合
    tf.keras.layers.Dense(10, activation="softmax"), # 输出层：10 类概率，和为 1
])

# ==================== ② Functional 函数式建模型（图式，安全默认） ====================
# 层是可调用对象：用张量接线构造计算图；支持多输入/多输出/分支合并(concat/add)/
# 共享层/残差——任何 Sequential 能做的它都能做。
inp = tf.keras.Input(shape=(28, 28))                # 图入口（不含 batch 维）
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inp, out)

# ==================== ③ compile / fit 三件套 + 回调 ====================
model.compile(optimizer="adam",                        # 字符串/实例均可，框架自动映射
              loss="sparse_categorical_crossentropy",  # 整数标签 0..9 → sparse 版（one-hot 用 categorical）
              metrics=["accuracy"])                    # 字符串自动映射

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,   # 或 validation_data=(x_val, y_val)，后者更可控
                    callbacks=[             # 把"断点/早停/降 lr/日志"从训练循环解耦出去
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=5,
                            restore_best_weights=True),   # 恢复最优权重，而非早停那一刻的权重
                        tf.keras.callbacks.ModelCheckpoint(
                            "best.keras", monitor="val_accuracy", save_best_only=True),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                        tf.keras.callbacks.TensorBoard(log_dir="logs"),
                    ],
                    shuffle=True)
# history 含逐 epoch 的 loss/acc

loss, acc = model.evaluate(x_test, y_test)  # evaluate 出 loss + metric
probs = model.predict(x_test)               # 推理 API，自动按 batch 循环（勿与 model(x) 混淆）
preds = probs.argmax(axis=1)                # 分类：概率 → 类别 0..9

# ==================== ④ 保存与复用 ====================
model.save("model.keras")                      # Keras 3 新格式；兼容老 "xxx.h5"
m2 = tf.keras.models.load_model("model.keras") # 重新加载即可继续 evaluate/predict/fit

# ==================== 案例详解：MNIST 手写数字识别 ====================
# 目标：28×28 灰度图 → 类别 0..9。
# 1) 数据：x_train shape=(60000, 28, 28)；y_train 为整数 0..9
#    → loss 必须用 sparse_categorical_crossentropy（sparse 版），否则报错/指标异常；
# 2) 模型：Flatten 把 (batch,28,28) 拉成 (batch,784)；Dense(128, relu) 提取特征；
#    Dropout(0.2) 训练时随机丢弃 20% 神经元抑制过拟合；末层 Dense(10, softmax) 输出 10 类概率；
# 3) 训练：compile 只"声明"adam + sparse 交叉熵 + accuracy，真正循环发生在 fit；
#    validation_split 从训练集末尾切 20% 做验证；EarlyStopping 监控 val_loss、patience=5，
#    配 restore_best_weights=True 恢复最优权重；ModelCheckpoint 只存 val_acc 最好的权重；
# 4) 评估：evaluate 得测试 loss/acc；predict 得到 (N,10) 概率矩阵，argmax(axis=1) 得到类别。
```

---
## 关联
- 前置：[[TensorFlow 基础]]——张量运算、batch 维约定、`tf.data` 数据管道是使用 Keras 的前提
- 前置：[[激活函数与损失函数]]——softmax + crossentropy 配对，数值稳定性由框架 fused 实现处理
- 类似：[[PyTorch 建模]]（区别是 PyTorch 以命令式 `nn.Module` + DataLoader + 手写训练循环为主，Keras 以配置式 compile/fit 声明训练为主；callback 概念对位 torch 的 scheduler + checkpoint）
- 进阶：[[TensorFlow 自定义训练循环]]（区别是放弃 fit 的自动封装，用 `tf.GradientTape` 手动记录梯度并应用 optimizer，换取逐 step 的细粒度控制；上层模型与层仍可复用）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| Sequential / Functional + compile/fit（本文方案） | 配置式声明训练：堆层/张量接线建图，compile 声明优化器/损失/指标，fit 自动跑批次循环与反向传播；回调把早停/断点/降 lr/日志解耦出去 | 标准监督学习快速原型：图像/表格分类、回归、多输入多输出，MNIST 级任务，五分钟跑通 |
| tf.GradientTape 手写训练循环（替代方案） | 命令式：用 tape 记录前向梯度，手动执行 loss → tape.gradient → optimizer.apply_gradients，逐 step 全控 | 需要自定义损失/梯度裁剪/特殊更新规则，compile/fit 封装不住的研究型训练代码 |
| PyTorch（nn.Module + DataLoader）（替代方案） | 面向对象层模块 + autograd，训练循环显式编写，动态图天然可调试 | 学术实验/研究原型、需要动态控制流与逐步调试、生态偏研究代码的场景 |

---
## 参考
- [Keras Sequential 模型官方指南](https://keras.io/guides/sequential_model/)
- [Keras Functional API 官方指南](https://keras.io/guides/functional_api/)
- [Keras 内置方法训练与评估（compile / fit / evaluate）](https://keras.io/guides/training_with_built_in_methods/)
- [Keras Callbacks API 文档](https://keras.io/api/callbacks/)

---
## 具体案例
- [[Keras 快速上手实战案例]](Keras快速上手_sample.py)
