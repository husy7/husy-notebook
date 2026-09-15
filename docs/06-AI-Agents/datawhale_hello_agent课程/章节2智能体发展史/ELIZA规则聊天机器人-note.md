---
title: "ELIZA 规则聊天机器人"
tags: [ai, agent, hello-agents, 智能体发展史, ELIZA]
date: 2026-09-14
---

# ELIZA 规则聊天机器人

> 用"关键词 → 正则分解 → 代词转换 → 模板重组"的规则游戏，在不理解语义的情况下制造共情假象——"看上去像智能，其实没有理解"。

## 核心原理和流程

> 简记：识-解-换-组。本质是把"自然语言理解"降维成一个可操作的**模式匹配 + 文本替换**游戏。

### 2.2.1 设计思想（Weizenbaum，1966，MIT）

- ELIZA 不是单一程序，而是一个可执行不同"脚本"的框架；最著名的脚本 **DOCTOR** 模仿罗杰斯学派的非指导性心理治疗师。
- **反射式对话**：从不正面回答、不提供信息，而是识别用户输入中的关键词，用预设转换规则把陈述转成开放式提问。例：用户说"我为我的男朋友感到难过" → 识别"我为……感到难过" → 回应"你为什么会为你的男朋友感到难过？"
- **设计意图（反讽）**：魏泽鲍姆并非要造出真正"理解"情感的智能体，恰恰相反，他想证明简单句式转换就能营造"智能/共情"假象。
- **ELIZA 效应**：结果出乎意料——很多交互者（包括他的秘书）对它产生情感依赖，深信它能理解自己。

### 2.2.2 四步算法流程

```text
用户输入
   │ ① 关键词识别与排序（按优先级，最高优先者命中）
   ▼
② 分解 DECOMPOSE：带通配符 * 的规则捕获句子其余部分
   ├─ 规则 * my *  ← "My mother is afraid of me"
   └─ 捕获 ["", "mother is afraid of me"]
   ▼
③ 代词转换 TRANSFORM_PRONOUNS：I→you · my→your · me→you …
   ▼
④ 重组 REASSEMBLE：从关联模板随机选一条填充（增加多样性）
   ▼
回应
```

源文伪代码：

```text
FUNCTION generate_response(user_input):
    words = SPLIT(user_input)                       // 1. 拆词
    best_rule = FIND_BEST_RULE(words)               // 2. 找最高优先级规则
    IF best_rule is NULL: RETURN generic_response() //    无规则 → "Please go on."
    decomposed = DECOMPOSE(input, best_rule.pattern)// 3. 分解
    IF failed: RETURN generic_response()
    transformed = TRANSFORM_PRONOUNS(decomposed)    // 4. 代词转换
    return REASSEMBLE(transformed, best_rule.patterns) // 5. 重组
```

### 2.2.3 核心实现（关键片段摘录 + 中文注释）

规则库：正则模式 → 响应模板列表，`{0}` 用捕获内容填充：

```python
# 规则库：模式(正则) -> 响应模板列表；遍历顺序即优先级，越靠前越先命中
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.*': [                          # 兜底通配：无特定规则命中时用
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}
```

代词转换：第一/第二人称互换，维持"以你视角"回应的连贯性：

```python
# 代词转换表：把用户视角换成治疗师视角
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were",
    "i'd": "you would", "i've": "you have", "i'll": "you will",
    "yours": "mine", "mine": "yours"
}

def swap_pronouns(phrase):
    """对短语中的代词做第一/第二人称转换"""
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)
```

响应函数 + 主聊天循环：

```python
def respond(user_input):
    """根据规则库生成响应"""
    for pattern, responses in rules.items():   # dict 顺序 == 优先级
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            captured_group = match.group(1) if match.groups() else ''
            swapped_group = swap_pronouns(captured_group)      # 代词转换
            response = random.choice(responses).format(swapped_group)  # 随机挑模板填充
            return response
    return random.choice(rules[r'.*'])          # 兜底通配规则

# 主聊天循环：无状态，每次只处理当前单句
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
```

## 易错点 / 局限（为什么"像智能其实没有理解"）

> **缺乏语义理解**：不理解词义。输入 `I am not happy` 仍命中 `I am (.*)`，输出语义不通的回应——它无法理解否定词 "not" 的作用。
> **无上下文记忆（无状态 Stateless）**：每次回应只基于当前单句输入，无法进行连贯多轮对话。
> **规则的扩展性问题（组合爆炸）**：加规则会让规则库规模爆炸式增长，规则间冲突与优先级管理极复杂，难以维护。
> **ELIZA 效应的来源**：智能幻觉来自巧妙的对话策略（扮演被动提问者、开放式模板）+ 人类天生的情感投射心理，而非真正的理解。

## 练习

- Q1：ELIZA 生成回应的四步流程？
  A1：关键词识别与排序 → 分解（通配符捕获）→ 代词转换 → 重组（模板随机填充）。
- Q2：为什么 `r'.*'` 通配规则要放在 rules 字典末尾？
  A2：代码按字典遍历顺序决定优先级；`.*` 能匹配一切，放最后保证特定规则优先命中，仅作兜底。
- Q3：给出"看上去智能、实则没理解"的一个具体证据。
  A3：`I am not happy` 会被机械套用 `I am (.*)` 模板生成不合理的"你为什么不 happy"式回应，说明它处理不了否定语义。
- Q4：能否用数学说明开放域对话的"组合爆炸"？
  A4：开放域语义组合数接近无限，规则数随"关键词 × 场景 × 句式"呈组合/指数式增长，穷举不完；规则间冲突也随数量上升而急剧增加。

## 知识关联

- 前置：[符号主义智能体](符号主义智能体-note.md)（"预编码规则"思想的直接落地）、[[模式匹配]]
- 横向：[心智社会](心智社会-note.md)（从"单一规则引擎"到"多单元协作"的反思起点）、[[自然语言处理]]
- 进阶：[学习范式演进](学习范式演进-note.md)（用"学习"取代"穷举规则"）、[本章小结](本章小结-note.md)

## 对比与选型

| 维度           | ELIZA（规则）       | LLM 对话（如 ChatGPT）         |
| -------------- | ------------------- | ------------------------------ |
| 语义理解       | 无                  | 有（统计 + 涌现）              |
| 状态           | 无状态              | 有上下文窗口                   |
| 扩展性         | 组合爆炸，难维护    | 泛化强                         |
| 可控 / 可解释  | 高（规则透明）      | 低（黑盒）                     |
| 开发成本       | 低、可完全预期      | 高、依赖模型与数据             |

**选型速查**：需强控话术、固定 SOP、可解释 → 规则型；开放域、需理解与泛化 → LLM。

## 参考

- 源文：第二章 智能体发展史 · 2.2（hello-agents docs/chapter2）