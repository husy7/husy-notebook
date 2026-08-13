---
title: "[双指针算法笔记]"
tags: [算法, 双指针，python]
date: 2026-08-11
---
- [一、双指针](#一双指针)
  - [1.1 什么](#11-什么)
  - [1.2 双指针的分类](#12-双指针的分类)
- [二、快慢指针（同向双指针）](#二快慢指针同向双指针)
  - [2.1 核心思想](#21-核心思想)
  - [2.2 通用模板](#22-通用模板)
  - [2.3 经典题型](#23-经典题型)
    - [题型1：删除有序数组重复项（保留1个）](#题型1删除有序数组重复项保留1个)
    - [题型2：删除有序数组重复项（保留2个）](#题型2删除有序数组重复项保留2个)
    - [题型3：移动零](#题型3移动零)
    - [题型4：移除指定元素](#题型4移除指定元素)
- [三、左右指针（相向双指针）](#三左右指针相向双指针)
  - [3.1 核心思想](#31-核心思想)
  - [3.2 通用模板](#32-通用模板)
  - [3.3 经典题型](#33-经典题型)
    - [题型1：两数之和（有序数组）](#题型1两数之和有序数组)
    - [题型2：回文判断](#题型2回文判断)
    - [题型3：三数之和](#题型3三数之和)
    - [题型4：盛最多水的容器](#题型4盛最多水的容器)
- [四、滑动窗口](#四滑动窗口)
  - [4.1 核心思想](#41-核心思想)
  - [4.2 通用模板](#42-通用模板)
  - [4.3 经典题型](#43-经典题型)
    - [题型1：无重复字符的最长子串](#题型1无重复字符的最长子串)
    - [题型2：长度最小的子数组](#题型2长度最小的子数组)
- [五、解题套路总结](#五解题套路总结)
  - [5.1 如何选择双指针类型](#51-如何选择双指针类型)
  - [5.2 关键判断条件](#52-关键判断条件)
  - [5.3 常见陷阱](#53-常见陷阱)
  - [5.4 复杂度分析](#54-复杂度分析)
- [六、练习建议](#六练习建议)
  - [6.1 必做题目（按顺序）](#61-必做题目按顺序)
  - [6.2 学习步骤](#62-学习步骤)


## 一、双指针

### 1.1 什么

双指针是一种在数组、链表等线性数据结构上使用的算法技巧，通过维护两个指针（索引）来协同完成任务。

### 1.2 双指针的分类

| 类型         | 指针移动方向         | 典型应用                       |
| ------------ | -------------------- | ------------------------------ |
| **快慢指针** | 同向移动             | 删除重复项、移动零、链表环检测 |
| **左右指针** | 相向移动             | 两数之和、回文判断、反转数组   |
| **滑动窗口** | 同向移动（维护区间） | 子数组问题、字符串匹配         |

---

## 二、快慢指针（同向双指针）

### 2.1 核心思想

- **快指针（fast）**：遍历整个数组，探索新元素
- **慢指针（slow）**：维护有效数组的边界，指向下一个要放置的位置
- **关键**：慢指针左边都是处理好的数据

### 2.2 通用模板

```python
def fast_slow_template(nums):
    if not nums:
        return 0

    slow = 0  # 或 1、2，根据题目而定

    for fast in range(len(nums)):
        # 判断条件根据题目变化
        if condition(nums[fast], nums[slow]):
            nums[slow] = nums[fast]
            slow += 1

    return slow  # 或 nums
```

### 2.3 经典题型

#### 题型1：删除有序数组重复项（保留1个）

```python
def removeDuplicates(nums):
    if not nums:
        return 0

    slow = 1  # 第一个元素一定保留
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow - 1]:  # 与慢指针前一个比较
            nums[slow] = nums[fast]
            slow += 1

    return slow

# 示例
# [1,1,2,2,3] -> [1,2,3,2,3], 返回3
```

**执行过程：**

```
初始: [1, 1, 2, 2, 3]
slow=1

fast=1: nums[1]=1 == nums[0]=1 → 跳过
fast=2: nums[2]=2 != nums[0]=1 → nums[1]=2, slow=2
        [1, 2, 2, 2, 3]
fast=3: nums[3]=2 == nums[1]=2 → 跳过
fast=4: nums[4]=3 != nums[1]=2 → nums[2]=3, slow=3
        [1, 2, 3, 2, 3]
返回3
```

#### 题型2：删除有序数组重复项（保留2个）

```python
def removeDuplicates2(nums):
    if len(nums) <= 2:
        return len(nums)

    slow = 2  # 前两个元素一定保留
    for fast in range(2, len(nums)):
        if nums[fast] != nums[slow - 2]:  # 与慢指针前两个比较
            nums[slow] = nums[fast]
            slow += 1

    return slow

# 示例
# [1,1,1,2,2,3] -> [1,1,2,2,3,3], 返回5
```

**执行过程：**

```
初始: [1, 1, 1, 2, 2, 3]
slow=2

fast=2: nums[2]=1 == nums[0]=1 → 跳过
fast=3: nums[3]=2 != nums[0]=1 → nums[2]=2, slow=3
        [1, 1, 2, 2, 2, 3]
fast=4: nums[4]=2 == nums[1]=1 → 跳过  # 注意：比较的是nums[1]=1
fast=5: nums[5]=3 != nums[2]=2 → nums[3]=3, slow=4
        [1, 1, 2, 3, 2, 3]
返回4
```

#### 题型3：移动零

```python
def moveZeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

# 示例
# [0,1,0,3,12] -> [1,3,12,0,0]
```

**执行过程：**

```
初始: [0, 1, 0, 3, 12]
slow=0

fast=0: nums[0]=0 → 跳过
fast=1: nums[1]=1 != 0 → 交换 nums[0]和nums[1]
        [1, 0, 0, 3, 12], slow=1
fast=2: nums[2]=0 → 跳过
fast=3: nums[3]=3 != 0 → 交换 nums[1]和nums[3]
        [1, 3, 0, 0, 12], slow=2
fast=4: nums[4]=12 != 0 → 交换 nums[2]和nums[4]
        [1, 3, 12, 0, 0], slow=3
```

#### 题型4：移除指定元素

```python
def removeElement(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow

# 示例
# [3,2,2,3], val=3 -> [2,2,3,3], 返回2
```

---

## 三、左右指针（相向双指针）

### 3.1 核心思想

- **左指针（left）**：从数组左端开始
- **右指针（right）**：从数组右端开始
- 根据条件决定移动左指针还是右指针，直到相遇

### 3.2 通用模板

```python
def two_pointer_template(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # 根据条件处理
        if condition_left:
            left += 1
        elif condition_right:
            right -= 1
        else:
            # 找到答案或执行操作
            left += 1
            right -= 1
```

### 3.3 经典题型

#### 题型1：两数之和（有序数组）

```python
def twoSum(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # 和太小，左指针右移
        else:
            right -= 1  # 和太大，右指针左移

    return [-1, -1]

# 示例
# nums=[2,7,11,15], target=9 -> [0,1]
```

**执行过程：**

```
nums=[2, 7, 11, 15], target=9
left=0, right=3

nums[0]+nums[3]=2+15=17 > 9 → right=2
nums[0]+nums[2]=2+11=13 > 9 → right=1
nums[0]+nums[1]=2+7=9 == 9 → 返回[0,1]
```

#### 题型2：回文判断

```python
def isPalindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        # 跳过非字母数字字符
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

#### 题型3：三数之和

```python
def threeSum(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue  # 跳过重复

        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

#### 题型4：盛最多水的容器

```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # 计算当前面积
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        # 移动较短的边
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

# 示例
# height=[1,8,6,2,5,4,8,3,7] -> 49
```

---

## 四、滑动窗口

### 4.1 核心思想

维护一个动态窗口，窗口大小可变，通过移动左右边界来寻找最优解。

### 4.2 通用模板

```python
def sliding_window(s):
    left = 0
    window = {}  # 或 collections.Counter()
    result = 0

    for right in range(len(s)):
        # 将 right 指向的元素加入窗口
        window[s[right]] = window.get(s[right], 0) + 1

        # 收缩窗口（根据条件）
        while condition_to_shrink:
            # 更新结果
            result = max(result, right - left + 1)
            # 移除 left 指向的元素
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # 更新结果（根据题目）
        result = max(result, right - left + 1)

    return result
```

### 4.3 经典题型

#### 题型1：无重复字符的最长子串

```python
def lengthOfLongestSubstring(s):
    left = 0
    char_set = set()
    max_len = 0

    for right in range(len(s)):
        # 如果字符重复，移动左指针直到没有重复
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len

# 示例
# s="abcabcbb" -> 3 ("abc")
```

#### 题型2：长度最小的子数组

```python
def minSubArrayLen(target, nums):
    left = 0
    current_sum = 0
    min_len = float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]

        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return 0 if min_len == float('inf') else min_len

# 示例
# target=7, nums=[2,3,1,2,4,3] -> 2 ([4,3])
```

---

## 五、解题套路总结

### 5.1 如何选择双指针类型

| 问题特征               | 推荐类型      | 例子               |
| ---------------------- | ------------- | ------------------ |
| 数组有序，需要删除重复 | 快慢指针      | 删除排序数组重复项 |
| 数组无序，需要移动元素 | 快慢指针      | 移动零             |
| 数组有序，找特定和     | 左右指针      | 两数之和           |
| 数组乱序，找特定组合   | 排序+左右指针 | 三数之和           |
| 字符串/数组子区间      | 滑动窗口      | 最长子串           |
| 数组中有重复，需去重   | 快慢指针      | 去除重复           |

### 5.2 关键判断条件

```python
# 删除重复（保留1个）
if nums[fast] != nums[slow - 1]:
    nums[slow] = nums[fast]
    slow += 1

# 删除重复（保留2个）
if nums[fast] != nums[slow - 2]:
    nums[slow] = nums[fast]
    slow += 1

# 移动零
if nums[fast] != 0:
    nums[slow], nums[fast] = nums[fast], nums[slow]
    slow += 1

# 移除元素
if nums[fast] != val:
    nums[slow] = nums[fast]
    slow += 1

# 两数之和
if nums[left] + nums[right] == target:
    return [left, right]
elif nums[left] + nums[right] < target:
    left += 1
else:
    right -= 1
```

### 5.3 常见陷阱

1. **越界问题**：检查指针是否在有效范围内
2. **重复处理**：跳过重复元素
3. **边界条件**：空数组、单元素数组
4. **原地修改**：不要创建新数组

### 5.4 复杂度分析

| 算法类型 | 时间复杂度 | 空间复杂度 |
| -------- | ---------- | ---------- |
| 快慢指针 | O(n)       | O(1)       |
| 左右指针 | O(n)       | O(1)       |
| 滑动窗口 | O(n)       | O(k)       |

---

## 六、练习建议

### 6.1 必做题目（按顺序）

**快慢指针：**

1. 26. 删除有序数组中的重复项
2. 80. 删除有序数组中的重复项 II
3. 283. 移动零
4. 27. 移除元素

**左右指针：** 5. 167. 两数之和 II - 输入有序数组 6. 125. 验证回文串 7. 15. 三数之和 8. 11. 盛最多水的容器

**滑动窗口：** 9. 3. 无重复字符的最长子串 10. 209. 长度最小的子数组 11. 76. 最小覆盖子串

### 6.2 学习步骤

1. **先理解模板**：记住每种类型的通用解法
2. **写伪代码**：理清思路后再编码
3. **画图理解**：用具体例子模拟指针移动
4. **分析边界**：考虑空数组、单元素等特殊情况
5. **优化代码**：减少冗余判断

记住：**双指针的精髓在于通过指针的移动，将 O(n²) 的问题优化到 O(n)**，核心是找到指针移动的规律！
