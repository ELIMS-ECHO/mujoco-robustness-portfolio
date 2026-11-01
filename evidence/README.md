# 实验原始证据说明

本目录保存了[30 个随机种子独立评估报告](../EVAL30_REPORT_2026-11-01.md)所使用的机器可读证据。

## 文件说明

- summary.csv：七组条件的精简汇总，可直接在 GitHub 页面预览。
- summary.json：评估脚本导出的完整条件汇总。
- episodes.csv.gz：原始 210 行逐回合 CSV 的 gzip 压缩文件。压缩仅用于减小 Git 仓库体积，没有对数据重新抽样或转换。
- model_provenance.json：上游模型标识、不可变 Hugging Face 版本和模型权重 SHA-256。

## 还原原始 CSV

使用 Python 3.11，在仓库根目录执行：

~~~powershell
python -c "import gzip, pathlib; p=pathlib.Path('evidence/episodes.csv.gz'); pathlib.Path('evidence/episodes.csv').write_bytes(gzip.decompress(p.read_bytes()))"
~~~

还原后，请使用以下 SHA-256 校验文件完整性：

4adea82b096b3f100a4d830b084a3bd46861ccd69c280d3e693a5c59a38c0dd1

压缩文件本身的 SHA-256：

e88cf4890a7eaaca2ce43698e53ea8da4ced5a25867ec87cbcabe9ee0b542f32

## 来源与限制

这 210 回合数据来自 2026-11-01 完成的一次 MuJoCo 评估，使用一个第三方 SAC+HER 预训练模型。公开仓库保留了原始逐回合 CSV 字节内容和汇总数据。模型来源清单保留不可变模型版本与权重 SHA-256；公开副本已省略原始运行器的私有路径。模型权重文件没有包含在本仓库中。上游模型卡未明确标注许可证，复用前应核实原作者的授权条款。

这 30 个随机种子的结果只代表一个固定模型在仿真环境中的表现，不能证明跨模型泛化，也不能证明真实机械臂性能。失败分类属于启发式诊断，不是经过验证的物理标签。
