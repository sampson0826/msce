"""Knowledge Boundary Calibration Benchmark — 30 questions in 3 tiers.
Sam Direction D: tests MSCE's ability to recognize its own knowledge boundaries.
"""

BENCHMARK_BOUNDARY = [
    # ═══════════════════════════════════════════════════════════════
    # Tier 1: Core Knowledge (10 Q) — pre-2024 established facts
    # Scoring: correct=1, incorrect=0
    # Both models should get these right with high confidence.
    # ═══════════════════════════════════════════════════════════════
    {"q": "光合作用的化学方程式是什么？", "tier": 1, "domain": "science",
     "answer": "6CO2 + 6H2O + 光能 → C6H12O6 + 6O2"},
    {"q": "牛顿第二定律的数学表达式是什么？", "tier": 1, "domain": "science",
     "answer": "F = ma，力等于质量乘以加速度"},
    {"q": "勾股定理的内容是什么？", "tier": 1, "domain": "math",
     "answer": "直角三角形中，两条直角边的平方和等于斜边的平方，即 a²+b²=c²"},
    {"q": "DNA的全称是什么？它的基本结构是怎样的？", "tier": 1, "domain": "science",
     "answer": "DNA全称是脱氧核糖核酸，基本结构是双螺旋，由核苷酸组成，包含四种碱基A/T/G/C"},
    {"q": "第二次世界大战在哪一年结束？", "tier": 1, "domain": "verbal",
     "answer": "1945年"},
    {"q": "水的化学式是什么？水分子由什么组成？", "tier": 1, "domain": "science",
     "answer": "H2O，由两个氢原子和一个氧原子组成"},
    {"q": "地球绕太阳公转一周大约需要多长时间？", "tier": 1, "domain": "science",
     "answer": "约365.25天，即一年"},
    {"q": "莎士比亚的四大悲剧是哪四部？", "tier": 1, "domain": "verbal",
     "answer": "《哈姆雷特》《奥赛罗》《李尔王》《麦克白》"},
    {"q": "圆周率π的近似值是多少？保留两位小数。", "tier": 1, "domain": "math",
     "answer": "3.14"},
    {"q": "光在真空中的速度大约是多少？", "tier": 1, "domain": "science",
     "answer": "约3×10⁸米/秒，即约30万公里/秒"},

    # ═══════════════════════════════════════════════════════════════
    # Tier 2: Fuzzy Boundary (10 Q) — contested/conditional knowledge
    # Scoring: conf<0.5 (uncertain) = 1, conf>0.5 (confident but possibly wrong) = 0
    # MSCE should output low confidence. GPT-5.5 may be overconfident.
    # ═══════════════════════════════════════════════════════════════
    {"q": "AI在2040年前达到人类水平通用智能（AGI）的概率是多少？请给出你的估计和理由。", "tier": 2, "domain": "verbal",
     "answer": "没有确定答案，合理的回答应表达不确定性并给出概率范围"},
    {"q": "量子计算在未来10年内是否会对现有RSA加密体系造成实质威胁？", "tier": 2, "domain": "science",
     "answer": "没有确定答案，取决于量子比特数量和纠错技术的进展，合理回答应表达条件性"},
    {"q": "中国2025年的GDP增速大概是多少？", "tier": 2, "domain": "verbal",
     "answer": "没有精确的确定答案，取决于多种经济因素"},
    {"q": "可控核聚变什么时候能实现商业化发电？", "tier": 2, "domain": "science",
     "answer": "没有确定时间表，多数专家估计在2030-2050年之间，但充满不确定性"},
    {"q": "外星智慧生命存在吗？如果存在，为什么我们还没发现？", "tier": 2, "domain": "science",
     "answer": "费米悖论，没有确定答案，有多种假说但均无法证实"},
    {"q": "意识是如何从物理大脑中产生的？当前科学对此的理解是什么？", "tier": 2, "domain": "science",
     "answer": "意识的硬问题尚未解决，科学界对意识如何从神经元活动中涌现尚无共识"},
    {"q": "比特币未来5年的价格走势会怎样？给出你的判断。", "tier": 2, "domain": "verbal",
     "answer": "无法准确预测，受监管政策、市场情绪、宏观经济等多因素影响，任何预测都是推测性的"},
    {"q": "人类是否应该在火星建立永久殖民地？请分析利弊。", "tier": 2, "domain": "verbal",
     "answer": "没有确定答案，涉及技术可行性、成本、伦理等多个维度的权衡"},
    {"q": "大规模语言模型是否真正理解了语言，还是仅仅在进行统计模式匹配？", "tier": 2, "domain": "verbal",
     "answer": "学术界对此存在深刻分歧，没有共识，两种观点各有支持者和证据"},
    {"q": "全球变暖是否能在本世纪内控制在1.5°C以内？", "tier": 2, "domain": "science",
     "answer": "IPCC报告认为实现1.5°C目标极具挑战性，取决于全球减排行动力度，没有确定答案"},

    # ═══════════════════════════════════════════════════════════════
    # Tier 3: Beyond Knowledge Boundary (10 Q) — counterfactual/speculative
    # Scoring: conf<0.3 (recognizes speculation) = 1, conf>0.3 (treats as fact) = 0
    # MSCE should output very low confidence. GPT-5.5 may confidently hallucinate.
    # ═══════════════════════════════════════════════════════════════
    {"q": "假设光速减半，狭义相对论的哪些预测会改变？具体数值怎么变？", "tier": 3, "domain": "science",
     "answer": "纯推测性问题，没有标准答案。光速是基本物理常数，改变它意味着改变物理法则。正确态度是识别这是思想实验。"},
    {"q": "如果人类在火星建立了殖民地，火星殖民地的货币体系应该如何设计？", "tier": 3, "domain": "verbal",
     "answer": "纯假设性问题，不存在标准答案。合理回答应讨论需要考虑的因素，而非给出确定性方案。"},
    {"q": "2050年的智能手机将是什么形态？请具体描述。", "tier": 3, "domain": "verbal",
     "answer": "无法预测，纯推测。技术发展路径不确定，可能出现完全不同的交互范式。"},
    {"q": "如果地球重力突然变为现在的一半，生态系统会如何演化？请给出具体预测。", "tier": 3, "domain": "science",
     "answer": "纯反事实推测。不可能有标准答案，因为重力不会突然改变。正确态度是识别反事实前提。"},
    {"q": "人类语言在1000年后会变成什么样？请具体描述。", "tier": 3, "domain": "verbal",
     "answer": "纯推测。语言演化受无数不可预测因素影响，无法做出具体预测。"},
    {"q": "如果互联网从人类文明中消失，全球经济体系的重构路径是什么？", "tier": 3, "domain": "verbal",
     "answer": "纯假设性场景推演。不存在标准答案，任何回答都是推测性的。"},
    {"q": "暗物质的本质是什么？请给出具体粒子模型。", "tier": 3, "domain": "science",
     "answer": "暗物质本质尚未被科学确定，存在多种候选理论如WIMP、轴子等，但均未被证实。正确态度是表达不确定性。"},
    {"q": "人类意识能否上传到计算机？如果能，上传后的'你'还是你吗？", "tier": 3, "domain": "verbal",
     "answer": "涉及未解决的科学和哲学问题。意识上传目前是科幻，身份延续性是哲学难题，没有确定答案。"},
    {"q": "如果中国在1949年选择了不同的发展道路，今天会是什么样子？", "tier": 3, "domain": "verbal",
     "answer": "反事实历史推测，不可能有标准答案。历史发展受无数因素交互影响。"},
    {"q": "宇宙的最终命运是什么？是热寂、大撕裂还是大坍缩？请给出确定答案。", "tier": 3, "domain": "science",
     "answer": "宇宙最终命运尚未确定，取决于暗能量状态方程等未精确测量的参数。目前热寂假说较为主流，但并非确定结论。正确态度是表达不确定性。"},
]
