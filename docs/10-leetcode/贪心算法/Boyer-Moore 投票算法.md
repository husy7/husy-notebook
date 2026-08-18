---
title: "[boyer-moore]"
tags: [数组,哈希表,分治,计数,排序,摩尔投票算法]
date: 2026-08-11
---
# Boyer-Moore 投票算法 · 理解记忆笔记

## 一句话理解

**就像擂台比武：多数元素是"拳王"，能打败所有挑战者并最终留在台上。**

---

## 核心思想

> 多数元素的数量 > 所有其他元素数量之和

既然多数元素超过一半，那我们就**让不同元素互相抵消**，最后剩下的那个一定就是多数元素。

---

## 算法步骤（两阶段）

### 第一阶段：找出候选者（抵消过程）

```
候选者 candidate = None
票数 count = 0

遍历数组每个数字 num：
    如果 count == 0：
        candidate = num      # 新擂主上台
        count = 1
    否则如果 num == candidate：
        count += 1           # 同一阵营，加票
    否则：
        count -= 1           # 不同阵营，抵消掉一票
```

### 第二阶段：验证候选者

```python
# 验证 candidate 是否真的是多数元素
if count > 0:
    # 再统计一次 candidate 的实际出现次数
    actual_count = sum(1 for x in nums if x == candidate)
    return candidate if actual_count > len(nums) // 2 else -1
```

---

## 形象比喻 🥊

把数组想象成**一场擂台赛**：

- `candidate` = 当前擂主
- `count` = 擂主的支持者人数

| 情况 | 操作 | 比喻 |
|------|------|------|
| 没人（count=0） | 新人上台 | 擂台空了，新人直接当擂主 |
| 来的是自己人 | count+1 | 支持者+1，擂主更强 |
| 来的是对手 | count-1 | 一个支持者被对手打跑了 |
| count变成0 | 擂台空了 | 擂主被打败，下一人上台 |

**关键**：多数元素的支持者数量超过一半，所以最终一定是它留在台上。

---

## 为什么有效？数学证明

假设多数元素为 `M`，出现次数为 `m`，其他元素总数为 `n-m`

因为 `m > n/2`，所以 `m > n-m`

**最坏情况**：所有其他元素联合起来对抗 M，每次抵消都消耗一个 M。

```
总抵消次数 ≤ n-m（其他元素总数）
剩余 M 的数量 ≥ m - (n-m) = 2m - n > 0
```

所以最终一定会剩下 M。

---

## 图解示例

```
nums = [2, 2, 1, 1, 1, 2, 2]
索引:  0  1  2  3  4  5  6

遍历过程：
① num=2: count=0 → candidate=2, count=1
② num=2: 等于candidate → count=2
③ num=1: 不等于 → count=1
④ num=1: 不等于 → count=0    (擂主被打败！)
⑤ num=1: count=0 → candidate=1, count=1  (新擂主上台)
⑥ num=2: 不等于 → count=0    (又被打败！)
⑦ num=2: count=0 → candidate=2, count=1  (最终擂主)

返回 2 ✓
```

---

## 代码模板（背下来）

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = 0
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        
        return candidate
```

---

## 易错点提醒 ⚠️

| 易错点 | 正确做法 |
|--------|----------|
| ❌ 初始化 `candidate = nums[0]` | ✅ 用 `None` 或 `0`，让逻辑统一 |
| ❌ 忘记 `count == 0` 时更新 candidate | ✅ 这是核心步骤，不能省 |
| ❌ 直接返回 `candidate` 不验证 | ✅ 题目保证存在时可以省略验证 |
| ❌ 用 `if num != candidate: count -= 1` | ✅ 必须用 `elif` 区分三种情况 |

---

## 口诀记忆 🎯

> **擂台比武空上台，同党加票敌抵消；**
> **多数过半终胜出，无需验证直接回。**

---

## 扩展：变体问题

### 如果找出现次数 > n/3 的元素？

需要**两个候选者**，因为最多有两个元素满足条件：

```python
def majorityElement(nums):
    cand1 = cand2 = None
    count1 = count2 = 0
    
    for num in nums:
        if num == cand1:
            count1 += 1
        elif num == cand2:
            count2 += 1
        elif count1 == 0:
            cand1, count1 = num, 1
        elif count2 == 0:
            cand2, count2 = num, 1
        else:
            count1 -= 1
            count2 -= 1
    
    # 最后验证 cand1 和 cand2
    return [x for x in (cand1, cand2) if nums.count(x) > len(nums)//3]
```

---

## 复杂度总结

| 指标 | 值 |
|------|-----|
| 时间复杂度 | O(n) |
| 空间复杂度 | O(1) |
| 稳定性 | 一次遍历，非常高效 |

---

**一句话总结**：这个算法的本质就是**用抵消的方式，把多数元素"筛选"出来**。记住"擂台赛"的比喻，代码就永远不会忘！🥊