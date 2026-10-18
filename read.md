# 客户流失预测：中文运行说明

这是一个独立项目，GitHub 仓库建议命名为 `customer-churn-prediction`。
根目录的 `README.md` 是英文项目介绍，本文件是中文使用说明。

## 如何运行

需要 Python 3.10 或更新版本。核心程序、实验和测试只使用标准库，不需要 NumPy 或 scikit-learn。
在这个文件所在的项目目录打开终端，运行：

```bash
python trees.py
python experiments.py
python -m unittest discover -s tests -v
```

- 第一条打印数据规模、类别分布和多数类基准。
- 第二条运行完整实验，包括 2.4.3 加权投票、2.4.4 提前停止；新结果写入 `results/`。
- 第三条运行自动化测试。
- 已运行的全部实验与文字解释保存在 `report.ipynb`，可以直接用 VS Code 打开查看。
- 如需使用 JupyterLab，可运行 `python -m pip install -r requirements-notebook.txt`，再运行 `python -m jupyterlab report.ipynb`。

## 模型包含什么

信息增益决策树、数值中点阈值搜索、Bagging、每个节点随机抽取特征的随机森林、根据验证集准确率加权投票的浅树集成，以及对小节点和低信息增益的提前停止。

随机主种子是 **3411**。类别特征与数值特征会在 2.3 及后面的实验一起使用。
所有准确率都与对应数据集的基准比较，并同时报告流失类 F1。
加权投票使用验证集设置权重，因此验证集成绩不能视为独立测试；测试集不会参与设置权重。

`data/` 已包含三个课程数据 CSV 和特征说明，方便完整运行。
数据路径由代码文件的位置决定，不能把三个 CSV 放到另一个 `data/data/` 子目录中。
项目保留了原始课程代码来源说明，见 `NOTICE.md`。

## 上传 GitHub

1. 在 GitHub 新建一个空仓库，例如 `customer-churn-prediction`。
2. 解压压缩包后，进入里面的同名项目文件夹。
3. 上传该文件夹内的文件与子文件夹，让 `README.md` 出现在仓库首页。
4. 确认 `data/`、`results/`、`tests/` 与 `trees.py` 在同一层。
5. 如使用 Git，在仓库目录依次运行 `git init`、`git add .`、`git commit -m "Add complete churn prediction project"`，再按 GitHub 空仓库页面给出的地址连接远程仓库并推送。

训练时间受电脑速度影响；提交的计时只是当前环境的一次测量。
