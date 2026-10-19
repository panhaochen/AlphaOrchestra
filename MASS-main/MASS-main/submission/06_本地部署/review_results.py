from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "05_技术证据"
RESULTS = EVIDENCE / "02_指标快照" / "主要结果.csv"

print("=" * 72)
print("AlphaOrchestra 离线评审入口")
print("=" * 72)
print("项目定位：多智能体投研编排与结果追溯平台")
print("主要结论：2024Q2 同期间控制中，增强系统的 5 日和 10 日排序指标优于 4 智能体控制组。")
print("效率：日均激活 14.14/64 个候选，较全量候选减少 77.9%。")
print("扩展检验：2026 扩展期 10 日 RankIC 0.011121，RankICIR 0.131179。")
print()

if RESULTS.exists():
    print(f"指标文件：{RESULTS}")
    with RESULTS.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows[:12]:
        print(" | ".join(row))
else:
    print("未找到指标快照，请检查 05_技术证据 是否与本目录同级。")

print()
print("建议核查顺序：")
print(f"1. {EVIDENCE / '00_证据索引.md'}")
print(f"2. {EVIDENCE / '03_验证报告'}")
print(f"3. {EVIDENCE / '01_运行产物' / 'q2_enhanced' / 'manifest.json'}")
print(f"4. {EVIDENCE / '01_运行产物' / 'q2_enhanced' / 'routing.parquet'}")
print(f"5. {EVIDENCE / '01_运行产物' / 'q2_enhanced' / 'signals.parquet'}")
print()
print("说明：历史回测不代表未来收益，系统当前用于辅助研究与模拟跟踪。")

