# -*- coding: utf-8 -*-
"""
智能 Agent 系统（Vercel 版 — 无服务端持久化）
使用 LLM 理解用户意图，提取参数，Python 执行计算
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from tools import (
    calculate_pension_contribution,
    calculate_medical_contribution,
    calculate_housing_fund_contribution,
    calculate_all_insurances,
    estimate_pension,
    PENSION_CONFIG,
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class InsuranceAgent:
    """五险一金智能 Agent"""

    def __init__(self, user_memory):
        self.user_memory = user_memory

    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        if not DEEPSEEK_API_KEY:
            return "系统错误：API 密钥未设置"

        url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 800
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                return ""
        except Exception as e:
            return f"LLM call failed: {str(e)}"

    def process(self, query: str) -> Tuple[Optional[str], bool]:
        """
        处理用户查询
        返回: (回答内容, 是否需要更多参数)
        如果返回 (None, False) 表示交给 RAG 处理
        """
        # LLM 分析意图（含地域检测）
        intent, params = self._analyze_intent(query)

        if params.get("is_non_shenzhen"):
            detected_city = params.get("non_shenzhen_city", "其他")
            return (
                f"您好，我是深圳市五险一金智能咨询助手，只解答深圳市社保公积金相关问题。\n\n"
                f"您提到的{detected_city}不属于深圳市范围，建议您拨打当地12333热线咨询。\n\n"
                f"如有深圳市社保公积金问题，欢迎继续问我。",
                False
            )

        if intent == "info_update":
            summary = self.user_memory.get_summary()
            return f"已更新您的个人信息。\n\n{summary}", False

        if intent == "calculation":
            result = self._execute_calculation(params)
            if result:
                return result, False
            return self._ask_for_params(params), True

        if intent == "info_query":
            return None, False

        return None, False

    def _analyze_intent(self, query: str) -> Tuple[str, Dict]:
        user_info = self._get_user_info_summary()

        prompt = f"""分析以下用户输入，判断用户意图并提取相关参数。

【用户输入】
{query}

【已知用户信息】
{user_info}

请分析并返回JSON格式结果：
{{
  "intent": "info_update | calculation | info_query | other",
  "params": {{
    "salary": 月工资数字(如有),
    "years": 缴费年限数字(如有),
    "housing_fund": 公积金金额数字(反向计算时用),
    "rate": 缴存比例(0.05-0.12，如有),
    "calculation_type": "pension | medical | housing | all | estimate_pension | housing_reverse",
    "missing_params": ["缺失的参数列表"],
    "is_non_shenzhen": true/false,
    "non_shenzhen_city": "检测到的非深圳城市名（如广州、北京），is_non_shenzhen为true时填写"
  }},
  "reasoning": "简短分析理由，20字以内"
}}

**意图判断规则**：

1. **info_update（信息更新）** - 用户声明或更正自己的信息
   - 特征：简短句、无疑问词、类似填表
   - 例子："我是深户"、"月薪8000"、"我退休了"、"累计交了10年"、"公积金按12%交"、"今年35岁"、"我是女的"
   - 注意："我60岁退休，到时候能拿多少"不是info_update，是calculation+info_query

2. **calculation（计算）** - 请求计算缴费金额
   - 特征：包含数字、询问金额、"交多少/要交多少"
   - 例子："我交多少社保"、"月薪15000元，养老保险多少"、"公积金5000元，反推工资多少"

3. **info_query（政策查询）** - 询问规定、条件、流程
   - 特征：疑问词(怎么/什么/多少/为什么/能否/几)、询问规则
   - 例子："医保怎么报销"、"提取公积金需要什么条件"、"深户和非深户有什么区别"

4. **other（其他）** - 闲聊、问候、无关内容
   - 例子："你好"、"谢谢"、"再见"

**地域检测规则**：
- 如果用户明确询问深圳以外城市的社保或公积金政策，设置 is_non_shenzhen=true 并填写 non_shenzhen_city
- 如果只是提及深圳（含"深圳"、"深圳市"、"深"等），设置 is_non_shenzhen=false
- 如果没有明确提到其他城市，默认 is_non_shenzhen=false

只返回JSON，不要其他内容。"""

        messages = [
            {"role": "system", "content": "你是信息提取助手，只返回JSON格式的分析结果。"},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages)

        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get("intent", "other")
                params = result.get("params", {})
                return intent, params
        except Exception:
            pass

        return "other", {}

    def _execute_calculation(self, params: Dict) -> Optional[str]:
        calc_type = params.get("calculation_type", "all")
        salary = params.get("salary") or self.user_memory.monthly_salary
        years = params.get("years") or self.user_memory.cumulative_years or 15
        rate = params.get("rate") or self.user_memory.housing_fund_rate or 0.05
        housing_fund = params.get("housing_fund")

        if calc_type == "housing_reverse" and housing_fund:
            if self.user_memory.monthly_salary > 0:
                salary = self.user_memory.monthly_salary
            else:
                salary = None

        if not salary and calc_type != "housing_reverse":
            return None

        results = []
        results.append(f"**基于您提供的信息计算**")
        results.append(f"(城市: 深圳 | 就业状态: {self.user_memory.employment_status or '未知'} | 户籍: {self.user_memory.household_type or '未知'})")
        if salary:
            results.append(f"月工资: {salary}元")
        results.append("")

        try:
            if calc_type == "pension":
                result = calculate_pension_contribution(salary)
                results.append("**养老保险缴费**")
                results.append(f"缴费基数：{result['缴费基数']} 元")
                results.append(f"个人缴费（8%）：{result['个人缴费']} 元/月")
                results.append(f"单位缴费（16%）：{result['单位缴费']} 元/月")
                results.append(f"合计：{result['总计']} 元/月")

            elif calc_type == "medical":
                level = self.user_memory.medical_level or "一档"
                result = calculate_medical_contribution(salary, level)
                results.append(f"**医疗保险缴费（{level}）**")
                if level == "一档":
                    results.append(f"个人缴费（2%）：{result['个人缴费']} 元/月")
                    results.append(f"单位缴费（6%）：{result['单位缴费']} 元/月")
                else:
                    results.append(f"个人缴费（0.5%）：{result['个人缴费']} 元/月")
                    results.append(f"单位缴费（1.5%）：{result['单位缴费']} 元/月")
                results.append(f"合计：{result['总计']} 元/月")

            elif calc_type == "housing":
                result = calculate_housing_fund_contribution(salary, rate)
                results.append("**住房公积金缴存**")
                results.append(f"缴存基数：{result['缴存基数']} 元")
                results.append(f"缴存比例：{result['缴存比例']}")
                results.append(f"个人缴存：{result['个人缴存']} 元/月")
                results.append(f"单位缴存：{result['单位缴存']} 元/月")
                results.append(f"合计：{result['总计']} 元/月")

            elif calc_type == "all":
                result = calculate_all_insurances(salary, salary, self.user_memory.medical_level or "一档", salary, rate)
                results.append("**五险一金计算结果**")
                for name in ["养老保险", "医疗保险", "失业保险", "生育保险", "工伤保险", "住房公积金"]:
                    if name in result and name != "汇总":
                        r = result[name]
                        if name == "住房公积金":
                            results.append(f"**{name}**：{r['个人缴存']} + {r['单位缴存']} = {r['总计']} 元/月")
                        else:
                            results.append(f"**{name}**：{r['个人缴费']} + {r['单位缴费']} = {r['总计']} 元/月")
                summary = result["汇总"]
                results.append("")
                results.append("**汇总**")
                results.append(f"个人总缴纳：{summary['个人总缴纳']} 元/月")
                results.append(f"单位总缴纳：{summary['单位总缴纳']} 元/月")
                results.append(f"**合计：{summary['总计']} 元/月**")

            elif calc_type == "estimate_pension":
                is_retired = self.user_memory.employment_status == "退休"
                age = self.user_memory.age or 30
                gender = self.user_memory.gender or "男"

                RETIREMENT_AGE = {"男": 60, "女": 50}
                retirement_age = RETIREMENT_AGE.get(gender, 60)

                if is_retired:
                    actual_years = self.user_memory.cumulative_years or years
                    results.append(f"**养老金估算**（您已退休，已累计缴费 {actual_years} 年）")
                    monthly_contrib = salary * PENSION_CONFIG["personal_rate"]
                    result = estimate_pension(monthly_contrib, int(actual_years))
                    if "error" in result:
                        results.append(f"⚠ {result['error']}")
                    else:
                        results.append(f"预估月养老金：{result['预估月养老金']} 元")
                        results.append(f"  - 基础养老金：{result['基础养老金']} 元")
                        results.append(f"  - 个人账户养老金：{result['个人账户养老金']} 元")
                else:
                    years_to_retirement = max(0, retirement_age - age)
                    total_estimated_years = years + years_to_retirement
                    results.append(f"**养老金预估**")
                    results.append(f"（您现在 {age} 岁，{gender}性，法定退休年龄 {retirement_age} 岁）")
                    results.append(f"当前累计缴费：{years} 年")
                    results.append(f"预计到退休还能再缴：{years_to_retirement} 年")
                    results.append(f"预计总缴费年限：{total_estimated_years} 年")
                    monthly_contrib = salary * PENSION_CONFIG["personal_rate"]
                    result = estimate_pension(monthly_contrib, int(total_estimated_years))
                    if "error" in result:
                        results.append(f"⚠ {result['error']}")
                    else:
                        results.append("")
                        results.append(f"**预估月养老金**：{result['预估月养老金']} 元")
                        results.append(f"  - 基础养老金：{result['基础养老金']} 元")
                        results.append(f"  - 个人账户养老金：{result['个人账户养老金']} 元")
                results.append(f"\n⚠ 以上为估算金额，实际金额以社保局核定为准")

            elif calc_type == "housing_reverse":
                if not housing_fund:
                    return None
                required_salary = housing_fund / (rate * 2)
                results.append("**反向推算工资**")
                results.append(f"已知公积金：{housing_fund} 元/月")
                results.append(f"缴存比例：{int(rate*100)}%")
                results.append(f"您的月工资大概需要：**{required_salary:.0f} 元**")
                results.append("")
                results.append(f"（按深圳市规定，公积金个人和单位各缴 {int(rate*100)}%，合计为工资的 {int(rate*200)}%）")

            else:
                return None

            return "\n".join(results)

        except Exception as e:
            return f"计算出错: {str(e)}"

    def _ask_for_params(self, params: Dict) -> str:
        missing = params.get("missing_params", [])
        if not missing:
            return "请告诉我您的月工资是多少？例如：月薪15000元"
        labels = {
            "salary": "月工资",
            "years": "累计缴费年限",
            "housing_fund": "公积金金额"
        }
        needed = [labels.get(m, m) for m in missing]
        return f"为了给您准确的计算，请告诉我：{', '.join(needed)}"

    def _get_user_info_summary(self) -> str:
        mem = self.user_memory
        parts = ["城市: 深圳（固定）"]
        if mem.employment_status:
            parts.append(f"就业状态: {mem.employment_status}")
        if mem.household_type:
            parts.append(f"户籍: {mem.household_type}")
        if mem.medical_level:
            parts.append(f"医保档次: {mem.medical_level}")
        if mem.monthly_salary > 0:
            parts.append(f"月工资: {mem.monthly_salary}元")
        if mem.cumulative_years > 0:
            parts.append(f"累计缴费年限: {mem.cumulative_years}年")
        if mem.housing_fund_rate > 0:
            parts.append(f"公积金缴存比例: {int(mem.housing_fund_rate*100)}%")
        return " | ".join(parts) if parts else "暂无信息"
