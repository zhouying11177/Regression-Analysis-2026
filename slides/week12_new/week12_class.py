# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["meta"]
# # 第12周：偏差-方差权衡与损失函数
#
# 今天不背定义，今天做三次“下注”：
#
# 1. 模型越复杂，真的越好吗？
# 2. 什么叫模型“太不稳定”？
# 3. 一个离群点，先刺穿 `RMSE` 还是 `MAE`？
#
# 课堂规则：
# - 先猜，再运行；
# - 先看图，再命名；
# - 先做判断，再听解释。

# %% [markdown] tags=["script", "teacher-only"] jupyter={"source_hidden": true}
# **Teacher note**
# - 开场只说“今天做三次下注”，不要先讲公式。
# - 每次 stage cell 前都停 10~20 秒，先让学生说预测。
# - 如果学生急着要定义，先压住，等图出来再命名。

# %% tags=["stage"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20260523)


# %% tags=["stage"]
def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.2 * x) + 0.15 * x


def make_noisy_sample(n=80, noise_std=0.35, x_low=-3.0, x_high=3.0, rng=None):
    rng = RNG if rng is None else rng
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


def polynomial_model(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def fit_predict_degree(x_train, y_train, x_eval, degree: int):
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


# %% [markdown] tags=["cue"]
# ## 第一幕：三位候选人，谁该上线？
#
# 现在有三个候选模型：
# - `degree = 1`
# - `degree = 4`
# - `degree = 15`
#
# **先下注，不准看后面的误差曲线：**
# - 谁最像欠拟合？
# - 谁最像过拟合？
# - 如果今天必须选一个上线，你先押谁？

# %% tags=["stage"]
x, y = make_noisy_sample(n=120, noise_std=0.35, rng=np.random.default_rng(7))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.35, random_state=42
)

x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
y_true_grid = true_function(x_grid.ravel())

degrees_to_show = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, degree in zip(axes, degrees_to_show):
    y_grid_pred, model = fit_predict_degree(x_train, y_train, x_grid, degree)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_rmse = root_mean_squared_error(y_train, train_pred)
    test_rmse = root_mean_squared_error(y_test, test_pred)

    ax.scatter(x_train[:, 0], y_train, s=18, alpha=0.6, label="train")
    ax.scatter(x_test[:, 0], y_test, s=18, alpha=0.6, label="test")
    ax.plot(
        x_grid[:, 0],
        y_true_grid,
        color="black",
        linewidth=2,
        linestyle="--",
        label="truth",
    )
    ax.plot(
        x_grid[:, 0],
        y_grid_pred,
        color="#d62728",
        linewidth=2.5,
        label=f"degree={degree}",
    )
    ax.set_title(
        f"degree={degree}\ntrain RMSE={train_rmse:.3f}, test RMSE={test_rmse:.3f}"
    )
    ax.set_xlabel("x")

axes[0].set_ylabel("y")
axes[-1].legend(loc="upper left", fontsize=10)
fig.suptitle("先只看三张图：谁最像会在真实世界翻车？", y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["checkpoint"]
# **Checkpoint 1**
#
# 只根据刚才那三张图，现在口头回答：
#
# > 如果你今天必须选一个模型上线，你会选 `1 / 4 / 15` 中的哪一个？为什么？

# %% [markdown] tags=["cue"]
# ## 揭晓：把 1 到 18 阶全部拉出来排位
#
# 现在不再看个别例子，而看完整成绩单。
#
# 先猜：
# - 训练误差会怎么走？
# - 测试误差会不会也一直往下？
# - “最优复杂度”大概会出现在很左边、很右边，还是中间？

# %% tags=["stage"]
records = []
for degree in range(1, 19):
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    records.append(
        {
            "degree": degree,
            "train_rmse": root_mean_squared_error(y_train, y_train_pred),
            "test_rmse": root_mean_squared_error(y_test, y_test_pred),
        }
    )

error_df = pd.DataFrame(records)
error_df["generalization_gap"] = error_df["test_rmse"] - error_df["train_rmse"]
error_df["rank_by_test"] = error_df["test_rmse"].rank(method="dense").astype(int)

best_degree = int(error_df.loc[error_df["test_rmse"].idxmin(), "degree"])
worst_gap_degree = int(error_df.loc[error_df["generalization_gap"].idxmax(), "degree"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    error_df["degree"],
    error_df["train_rmse"],
    marker="o",
    linewidth=2.2,
    label="train RMSE",
)
ax.plot(
    error_df["degree"],
    error_df["test_rmse"],
    marker="o",
    linewidth=2.2,
    label="test RMSE",
)
ax.axvline(
    best_degree,
    color="gray",
    linestyle="--",
    alpha=0.75,
    label=f"best degree={best_degree}",
)
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("RMSE")
ax.set_title("完整成绩单：训练误差 vs 测试误差")
ax.legend()
plt.tight_layout()
plt.show()

scoreboard = (
    error_df.sort_values("test_rmse")
    .loc[:, ["degree", "train_rmse", "test_rmse", "generalization_gap", "rank_by_test"]]
    .head(6)
    .reset_index(drop=True)
)
scoreboard

print(f"Best test RMSE occurs at degree = {best_degree}")
print(f"Largest generalization gap occurs at degree = {worst_gap_degree}")

# %% [markdown] tags=["explain"]
# **解释（现在再命名）**
#
# - 训练误差几乎一路下降；
# - 测试误差不是一路下降，而是先降后升；
# - 所以真正的问题不是“谁拟合得更狠”，而是“谁在新数据上更稳”。
#
# 这正是 **bias-variance tradeoff** 的舞台入口。

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note**
# - 这里要明确点破：训练误差最低 ≠ 最佳模型。
# - 如果学生说“那就直接交叉验证选最优 degree”，先肯定，再追问：
#   - “为什么会出现这条 U 形测试误差曲线？”
# - 把讨论引向 bias / variance，而不是停在调参技巧。

# %% [markdown] tags=["cue"]
# ## 第二幕：谁更“抖”？
#
# 下面不改真实规律，只反复换训练样本。
#
# **继续下注：**
# - `degree = 2` 和 `degree = 15`，谁的拟合曲线更不稳定？
# - 如果同一个问题下，每次换数据就学出不同答案，你会把这种风险叫做什么？

# %% tags=["stage"]
x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
y_eval_true = true_function(x_eval.ravel())

degree_predictions = {}
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, degree in zip(axes, [2, 15]):
    collected_preds = []
    for sample_idx in range(14):
        x_sample, y_sample = make_noisy_sample(
            n=35,
            noise_std=0.35,
            rng=np.random.default_rng(1000 + sample_idx),
        )
        y_pred, _ = fit_predict_degree(x_sample, y_sample, x_eval, degree)
        collected_preds.append(y_pred)
        ax.plot(x_eval[:, 0], y_pred, alpha=0.30, linewidth=1.4)

    degree_predictions[degree] = np.vstack(collected_preds)
    ax.plot(
        x_eval[:, 0],
        y_eval_true,
        color="black",
        linewidth=3,
        linestyle="--",
        label="truth",
    )
    ax.set_title(f"Repeated fits with degree={degree}")
    ax.set_xlabel("x")
    ax.legend(loc="upper left")

axes[0].set_ylabel("predicted y")
fig.suptitle("同一真实规律、不同训练样本：谁更抖？", y=1.05)
plt.tight_layout()
plt.show()

variance_summary = []
for degree, preds in degree_predictions.items():
    pointwise_std = preds.std(axis=0)
    variance_summary.append(
        {
            "degree": degree,
            "mean_prediction_std": pointwise_std.mean(),
            "max_prediction_std": pointwise_std.max(),
        }
    )

variance_df = pd.DataFrame(variance_summary)
variance_df

# %% [markdown] tags=["explain"]
# **解释**
#
# - 左边更稳，右边更抖；
# - “抖”不是口语而已，它意味着：训练样本一换，模型答案就明显变；
# - 这就是 **high variance** 的直观图像。

# %% [markdown] tags=["checkpoint"]
# **Checkpoint 2**
#
# 请补全：
#
# > high variance model 的危险，不是它不会拟合训练集，而是它对 ______ 过于敏感。

# %% [markdown] tags=["cue"]
# ## 第三幕：一个离群点，先刺穿谁？
#
# 现在我们只加一个很大的错误。
#
# **下注：**
# - `RMSE` 和 `MAE`，谁先“爆表”？
# - 如果你做的是预算预测 / 风控 / 医疗告警，哪个指标会更符合你的风险直觉？

# %% tags=["stage"]
y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
y_pred_outlier = y_pred_clean.copy()
y_pred_outlier[-1] = 80

metrics_df = pd.DataFrame(
    {
        "scenario": ["clean prediction", "one large outlier"],
        "RMSE": [
            root_mean_squared_error(y_true, y_pred_clean),
            root_mean_squared_error(y_true, y_pred_outlier),
        ],
        "MAE": [
            mean_absolute_error(y_true, y_pred_clean),
            mean_absolute_error(y_true, y_pred_outlier),
        ],
    }
)
metrics_df["RMSE/MAE ratio"] = metrics_df["RMSE"] / metrics_df["MAE"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
axes[0].scatter(range(len(y_true)), y_true, s=85, label="true")
axes[0].scatter(range(len(y_true)), y_pred_outlier, s=85, label="pred")
axes[0].set_title("One large outlier changes one prediction")
axes[0].set_xlabel("sample index")
axes[0].set_ylabel("value")
axes[0].legend()

metric_plot_df = metrics_df.melt(
    id_vars="scenario",
    value_vars=["RMSE", "MAE"],
    var_name="metric",
    value_name="value",
)
sns.barplot(data=metric_plot_df, x="scenario", y="value", hue="metric", ax=axes[1])
axes[1].set_title("谁被离群点刺得更重？")
axes[1].set_xlabel("")
axes[1].set_ylabel("metric value")
plt.tight_layout()
plt.show()

metrics_df

# %% [markdown] tags=["explain"]
# **解释**
#
# - `RMSE` 会因为平方惩罚而被大错迅速拉高；
# - `MAE` 也会上升，但不会被单个极端错误那么强烈地主导；
# - 所以指标不是“哪个好看”，而是在表达你的风险偏好。

# %% [markdown] tags=["checkpoint"]
# **Checkpoint 3**
#
# 如果你是业务负责人，请用一句话回答：
#
# > 什么时候我会更想看 `RMSE`，什么时候我会更想看 `MAE`？

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note**
# - 这里不要让学生只停留在“RMSE 更大”。
# - 要追问：
#   - “为什么更大？”
#   - “更大代表哪种风险被放大了？”
#   - “如果线上不能接受偶发大错，你会看哪个指标？”

# %% [markdown] tags=["meta"]
# ## 收束：今天真正该带走的不是公式，而是判断
#
# 三个判断：
#
# 1. **训练误差下降，不代表泛化能力更好。**
# 2. **模型太复杂时，危险常常来自 high variance。**
# 3. **`RMSE` 和 `MAE` 代表的是不同的代价函数与风险偏好。**
#
# 下一讲自然问题：
#
# > 如果高复杂度会带来 high variance，我们能不能主动给模型“加约束”？
#
# 这就进入 `Ridge / Lasso / Elastic Net`。

# %% [markdown] tags=["backup"]
# ## Backup
#
# 可选延伸：
# - 改变训练样本量，观察 variance 是否下降；
# - 改变噪声强度，观察最佳复杂度是否移动；
# - 一次加入多个 outliers，观察 `RMSE` / `MAE` 差距是否进一步扩大。
