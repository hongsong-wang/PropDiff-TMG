import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# 读取 CSV 文件
pred_df = pd.read_csv("/home/daibingxuan/workspace/microstructure_generation_3d/text_results/predicted_results.csv")
true_df = pd.read_csv("/home/daibingxuan/workspace/material_generation_LLM/data/datasets/properties.csv")

# 输出路径
output_dir = "comparison_plots"
os.makedirs(output_dir, exist_ok=True)

# 三个物理属性名称
attributes = ["Phi", "E", "Anisotropy"]
# attributes1 = ["Predicted_Phi", "Predicted_1/E", "Predicted_Anisotropy"]
colors =["#71b7ed", "#84c3b7", "#b8aeeb"]
true_df["E"] = 1.0 / true_df["E"]

for idx, attr in enumerate(attributes):
    pred = pred_df[attr].values.reshape(-1, 1)
    true = true_df[attr].values.reshape(-1, 1)
    print((pred-true).mean())
    # 拟合直线
    reg = LinearRegression().fit(true, pred)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r2 = reg.score(true, pred)

    # 生成拟合线
    x_range = np.linspace(true.min(), true.max(), 100).reshape(-1, 1)
    y_fit = reg.predict(x_range)

    # 画图
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=true.flatten(), y=pred.flatten(), label="Data", color=colors[idx])
    plt.plot(x_range, y_fit, color="black", label=f"Fit: y={slope:.2f}x+{intercept:.2f}, $R^2$={r2:.3f}")
    plt.xlabel(f"True {attr}")
    plt.ylabel(f"Predicted {attr}")
    plt.title(f"Prediction vs True: {attr}")
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attr.replace('/', '_')}_comparison_base1.png"), dpi=300)
    plt.close()

print(f"所有对比图已保存在 {output_dir} 文件夹中")
