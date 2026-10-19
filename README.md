# AlphaOrchestra

AlphaOrchestra 是一个面向投资组合构建的多智能体市场分歧分析与实验项目，基于 MASS（Multi-Agent Simulation Scaling）扩展，包含异构智能体、动态路由、分歧诊断、条件辩论、信号聚合与回测验证。

## 最终提交位置

本机最终提交目录为：

```text
D:\Desktop\codex\MASS\MASS-main\MASS-main\submission
```

在本仓库中的相对路径为 [`MASS-main/MASS-main/submission`](MASS-main/MASS-main/submission)。该目录包含统一信息表、项目说明、展示稿、技术证据、验证报告和本地部署包；评审或归档请以该目录内容为准。

> GitHub 不接受单文件超过 100 MB。仓库排除了可重复生成的 `submission.zip` 以及两份超过该限制的 `daily_history.parquet`；其余提交材料与数据均保留。需要该原始日行情文件时，请按下文重新下载数据。

## 运行环境

- Windows 10/11（项目附带 PowerShell 启动脚本）
- Conda 或 Miniconda
- Python 3.10
- 建议至少 16 GB 内存；大规模默认实验可能需要更多内存
- 联网复现需要可访问的 OpenAI 兼容模型 API

推荐从最终提交内的项目源码目录运行：

```powershell
cd .\MASS-main\MASS-main\submission\06_本地部署\项目源码
conda create -n alphaorchestra python=3.10 -y
conda activate alphaorchestra
pip install pdm
pdm install
```

也可使用 `pip` 安装评审所需的最小依赖：

```powershell
pip install -r ..\requirements_review.txt
```

## 如何获得数据

最终提交包已经包含用于复现与审阅的 CSI 300 示例数据、指标快照和典型运行产物。免费数据来源主要为 BaoStock，基准代码为 `sh.000300`。

如需重新下载沪深 300 基准行情，在项目源码目录运行：

```powershell
python .\scripts\download_csi300_benchmark.py --start 2018-01-01 --end 2026-09-30
```

输出默认写入：

```text
stock_disagreement/dataset/csi300/benchmark_000300.parquet
```

若要接入自己的数据，请将数据整理到 `stock_disagreement/dataset/`，并根据数据结构调整 `stock_disagreement/agent/basic_agent.py` 中的数据模态和加载逻辑。免费公开数据在历史成分、复权、基本面覆盖和点时一致性方面存在限制，不应直接替代生产级商业数据。

## 如何开始运行

### 1. 离线审阅（推荐先运行）

无需模型 API。在 `MASS-main/MASS-main` 目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\submission\06_本地部署\start_review.ps1
```

该命令读取随包提供的结果快照、验证报告和运行产物，不会访问网络，也不会重新调用大模型。

### 2. 配置联网复现

在 `submission/06_本地部署/项目源码` 中复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```text
MASS_LLM_BASE_URL=https://your-provider.example/v1
MASS_LLM_API_KEY=your-key
MASS_CHAT_MODEL=your-chat-model
MASS_REASONING_MODEL=your-reasoning-model
MASS_LLM_CACHE_MODE=readwrite
```

不要提交真实 API 密钥；仓库已忽略 `.env`。

检查联网环境：

```powershell
cd ..
powershell -ExecutionPolicy Bypass -File .\start_review.ps1 -Online
```

### 3. 运行短期示例

回到 `项目源码` 目录后运行：

```powershell
python -m stock_disagreement.main --random_seed 42 --num_investor_type 16 --num_agents_per_investor 4 --stock_pool csi_300 --stock_num 50 --selected_stock_num 3 --start_date 20240102 --end_date 20240105 --evaluation_start_date 20240102 --no-use_agent_distribution_modification --use_heterogeneous_agent_pool --use_adaptive_router --router_min_k 8 --router_max_k 24 --use_disagreement_diagnosis --use_contextual_aggregator --use_conditional_debate --max_agent_calls_per_date 24 --max_debate_calls_per_date 4 --run_id reviewer_smoke_seed42
```

更完整的参数、实验说明和已知限制参见 [`部署说明.md`](MASS-main/MASS-main/submission/06_本地部署/部署说明.md) 与最终提交中的项目文档。

## 免责声明

本项目用于研究、观点冲突分析与模拟回测，不构成投资建议。
