# -*- coding: utf-8 -*-
"""
Tool/Function Calling 系统 - 五险一金计算器
解决LLM数学不准确的问题，所有计算由Python完成
"""

from typing import Dict


# ============================================
# 2026年深圳市最新政策参数
# ============================================

# 养老保险参数
PENSION_CONFIG = {
    "min_contribution_base": 2360,     # 缴费基数下限（暂沿用2025年标准）
    "max_contribution_base": 27549,    # 缴费基数上限
    "min_years": 15,                   # 最低缴费年限
    "personal_rate": 0.08,             # 个人缴费比例
    "unit_rate": 0.16,                # 单位缴费比例
    "levels": {                         # 养老金档次
        "一档": "深户",
        "二档": "非深户",
        "三档": "非深户"
    }
}

# 医疗保险参数 (2026年)
MEDICAL_CONFIG = {
    "min_contribution_base": 6727,      # 缴费基数下限
    "max_contribution_base": 33633,     # 缴费基数上限
    "rate_level1_employee": 0.02,      # 一档个人缴费比例2%
    "rate_level1_unit": 0.06,          # 一档单位缴费比例6%（2026年恢复）
    "rate_level2_employee": 0.005,     # 二档个人缴费比例0.5%
    "rate_level2_unit": 0.015,         # 二档单位缴费比例1.5%
    "flexible_rate": 0.08,             # 灵活就业人员缴费比例8%
}

# 失业保险参数
UNEMPLOYMENT_CONFIG = {
    "min_contribution_base": 5510,     # 缴费基数下限（沿用2025年）
    "max_contribution_base": 27549,    # 缴费基数上限
    "employee_rate": 0.005,            # 个人缴费比例0.5%
    "unit_rate": 0.008,               # 单位缴费比例0.8%
}

# 生育保险参数
MATERNITY_CONFIG = {
    "unit_rate": 0.005,                # 单位缴费比例0.5%
}

# 工伤保险参数
WORK_INJURY_CONFIG = {
    # 根据行业风险类别确定费率，这里用简化值
    "average_rate": 0.004,             # 平均基准费率
}

# 住房公积金参数
HOUSING_FUND_CONFIG = {
    "min_rate": 0.05,                 # 最低缴存比例5%
    "max_rate": 0.12,                 # 最高缴存比例12%
    "min_base": 2360,                 # 最低缴存基数
    "max_base": 41190,                # 最高缴存基数（3倍平均工资）
}


# ============================================
# 计算工具函数
# ============================================

def calculate_pension_contribution(base: float, unit_rate: float = None, personal_rate: float = None) -> Dict:
    """
    计算养老保险缴费金额
    LLM只负责提取参数，精确计算由Python完成
    """
    if base <= 0:
        return {"error": "缴费基数必须大于0"}

    actual_base = max(PENSION_CONFIG["min_contribution_base"],
                      min(base, PENSION_CONFIG["max_contribution_base"]))

    personal = actual_base * PENSION_CONFIG["personal_rate"]
    unit = actual_base * PENSION_CONFIG["unit_rate"]

    return {
        "缴费基数": actual_base,
        "个人缴费": round(personal, 2),
        "单位缴费": round(unit, 2),
        "总计": round(personal + unit, 2),
        "备注": f"缴费基数范围: {PENSION_CONFIG['min_contribution_base']}-{PENSION_CONFIG['max_contribution_base']}"
    }


def calculate_medical_contribution(base: float, level: str = "一档") -> Dict:
    """
    计算医疗保险缴费金额
    """
    if base <= 0:
        return {"error": "缴费基数必须大于0"}

    actual_base = max(MEDICAL_CONFIG["min_contribution_base"],
                      min(base, MEDICAL_CONFIG["max_contribution_base"]))

    if level == "一档":
        personal = actual_base * MEDICAL_CONFIG["rate_level1_employee"]
        unit = actual_base * MEDICAL_CONFIG["rate_level1_unit"]
    else:  # 二档
        personal = actual_base * MEDICAL_CONFIG["rate_level2_employee"]
        unit = actual_base * MEDICAL_CONFIG["rate_level2_unit"]

    return {
        "医保档次": level,
        "缴费基数": actual_base,
        "个人缴费": round(personal, 2),
        "单位缴费": round(unit, 2),
        "总计": round(personal + unit, 2),
        "备注": f"{level}医保，基数范围: {MEDICAL_CONFIG['min_contribution_base']}-{MEDICAL_CONFIG['max_contribution_base']}"
    }


def calculate_unemployment_contribution(base: float) -> Dict:
    """
    计算失业保险缴费金额
    """
    if base <= 0:
        return {"error": "缴费基数必须大于0"}

    actual_base = max(UNEMPLOYMENT_CONFIG["min_contribution_base"],
                      min(base, UNEMPLOYMENT_CONFIG["max_contribution_base"]))

    personal = actual_base * UNEMPLOYMENT_CONFIG["employee_rate"]
    unit = actual_base * UNEMPLOYMENT_CONFIG["unit_rate"]

    return {
        "缴费基数": actual_base,
        "个人缴费": round(personal, 2),
        "单位缴费": round(unit, 2),
        "总计": round(personal + unit, 2),
        "备注": f"基数范围: {UNEMPLOYMENT_CONFIG['min_contribution_base']}-{UNEMPLOYMENT_CONFIG['max_contribution_base']}"
    }


def calculate_maternity_contribution(base: float) -> Dict:
    """
    计算生育保险缴费金额（单位缴纳，个人不缴）
    """
    if base <= 0:
        return {"error": "缴费基数必须大于0"}

    unit = base * MATERNITY_CONFIG["unit_rate"]

    return {
        "缴费基数": base,
        "单位缴费": round(unit, 2),
        "个人缴费": 0,
        "总计": round(unit, 2),
        "备注": "生育保险由单位缴纳，个人不缴费"
    }


def calculate_work_injury_contribution(base: float, industry_rate: float = None) -> Dict:
    """
    计算工伤保险缴费金额
    """
    if base <= 0:
        return {"error": "缴费基数必须大于0"}

    rate = industry_rate if industry_rate else WORK_INJURY_CONFIG["average_rate"]
    unit = base * rate

    return {
        "缴费基数": base,
        "单位缴费": round(unit, 4),
        "个人缴费": 0,
        "费率": f"{rate*100}%",
        "总计": round(unit, 2),
        "备注": "工伤保险由单位缴纳，个人不缴费"
    }


def calculate_housing_fund_contribution(base: float, rate: float = 0.05) -> Dict:
    """
    计算住房公积金缴存金额
    """
    if base <= 0:
        return {"error": "缴存基数必须大于0"}

    actual_base = max(HOUSING_FUND_CONFIG["min_base"],
                      min(base, HOUSING_FUND_CONFIG["max_base"]))

    actual_rate = max(HOUSING_FUND_CONFIG["min_rate"],
                      min(rate, HOUSING_FUND_CONFIG["max_rate"]))

    personal = actual_base * actual_rate
    unit = actual_base * actual_rate

    return {
        "缴存基数": actual_base,
        "缴存比例": f"{actual_rate*100}%",
        "个人缴存": round(personal, 2),
        "单位缴存": round(unit, 2),
        "总计": round(personal + unit, 2),
        "备注": f"比例范围: {HOUSING_FUND_CONFIG['min_rate']*100}%-{HOUSING_FUND_CONFIG['max_rate']*100}%"
    }


def calculate_all_insurances(
    pension_base: float,
    medical_base: float,
    medical_level: str = "一档",
    housing_base: float = None,
    housing_rate: float = 0.05
) -> Dict:
    """
    计算五险一金总额
    """
    results = {
        "养老保险": calculate_pension_contribution(pension_base),
        "医疗保险": calculate_medical_contribution(medical_base, medical_level),
        "失业保险": calculate_unemployment_contribution(medical_base),  # 失业基数同医疗
        "生育保险": calculate_maternity_contribution(medical_base),
        "工伤保险": calculate_work_injury_contribution(medical_base),
    }

    if housing_base:
        results["住房公积金"] = calculate_housing_fund_contribution(housing_base, housing_rate)

    # 计算总计（公积金使用"缴存" key，需要兼容处理）
    total_personal = sum(
        v.get("个人缴费", v.get("个人缴存", 0)) if isinstance(v, dict) else 0
        for v in results.values()
    )
    total_unit = sum(
        v.get("单位缴费", v.get("单位缴存", 0)) if isinstance(v, dict) else 0
        for v in results.values()
    )
    total_all = sum(
        v.get("总计", 0) if isinstance(v, dict) else 0
        for v in results.values()
    )

    results["汇总"] = {
        "个人总缴纳": round(total_personal, 2),
        "单位总缴纳": round(total_unit, 2),
        "总计": round(total_all, 2)
    }

    return results


def estimate_pension(personal_contribution: float, years: int, avg_salary: float = 177057) -> Dict:
    """
    估算养老金（简化估算）
    """
    if years < PENSION_CONFIG["min_years"]:
        return {
            "error": f"缴费年限不足，需要至少{PENSION_CONFIG['min_years']}年",
            "当前年限": years
        }

    # 简化计算：基础养老金 + 个人账户养老金
    base_pension = avg_salary / 12 * 0.2  # 假设按平均工资20%计算基础养老金
    personal_pension = personal_contribution * 12 * years / 139  # 计发月数139

    monthly_pension = base_pension + personal_pension

    return {
        "缴费年限": years,
        "预估月养老金": round(monthly_pension, 2),
        "基础养老金": round(base_pension, 2),
        "个人账户养老金": round(personal_pension, 2),
        "备注": "此为估算值，实际养老金以社保局计算为准"
    }


# ============================================
# 工具注册表 - LLM可调用的工具列表
# ============================================

TOOLS = [
    {
        "name": "calculate_pension",
        "description": "计算养老保险缴费金额，输入缴费基数",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴费基数"}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_medical",
        "description": "计算医疗保险缴费金额，输入缴费基数和档次(一档/二档)",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴费基数"},
                "level": {"type": "string", "description": "医保档次：一档或二档", "default": "一档"}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_unemployment",
        "description": "计算失业保险缴费金额，输入缴费基数",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴费基数"}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_maternity",
        "description": "计算生育保险缴费金额，输入缴费基数",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴费基数"}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_work_injury",
        "description": "计算工伤保险缴费金额，输入缴费基数",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴费基数"}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_housing_fund",
        "description": "计算住房公积金缴存金额，输入缴存基数和比例",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "缴存基数"},
                "rate": {"type": "number", "description": "缴存比例(0.05-0.12)", "default": 0.05}
            },
            "required": ["base"]
        }
    },
    {
        "name": "calculate_all",
        "description": "一次性计算五险一金全部金额",
        "parameters": {
            "type": "object",
            "properties": {
                "pension_base": {"type": "number", "description": "养老保险缴费基数"},
                "medical_base": {"type": "number", "description": "医疗保险缴费基数"},
                "medical_level": {"type": "string", "description": "医保档次", "default": "一档"},
                "housing_base": {"type": "number", "description": "住房公积金缴存基数"},
                "housing_rate": {"type": "number", "description": "公积金缴存比例", "default": 0.05}
            },
            "required": ["pension_base", "medical_base"]
        }
    },
    {
        "name": "estimate_pension",
        "description": "估算退休后每月养老金",
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_contribution": {"type": "number", "description": "每月个人缴费金额"},
                "years": {"type": "number", "description": "累计缴费年限"}
            },
            "required": ["monthly_contribution", "years"]
        }
    }
]


def call_tool(tool_name: str, params: Dict) -> Dict:
    """调用工具"""
    tool_map = {
        "calculate_pension": lambda p: calculate_pension_contribution(p.get("base", 0)),
        "calculate_medical": lambda p: calculate_medical_contribution(p.get("base", 0), p.get("level", "一档")),
        "calculate_unemployment": lambda p: calculate_unemployment_contribution(p.get("base", 0)),
        "calculate_maternity": lambda p: calculate_maternity_contribution(p.get("base", 0)),
        "calculate_work_injury": lambda p: calculate_work_injury_contribution(p.get("base", 0)),
        "calculate_housing_fund": lambda p: calculate_housing_fund_contribution(p.get("base", 0), p.get("rate", 0.05)),
        "calculate_all": lambda p: calculate_all_insurances(
            p.get("pension_base", 0),
            p.get("medical_base", 0),
            p.get("medical_level", "一档"),
            p.get("housing_base"),
            p.get("housing_rate", 0.05)
        ),
        "estimate_pension": lambda p: estimate_pension(p.get("monthly_contribution", 0), p.get("years", 0))
    }

    if tool_name in tool_map:
        try:
            return {"success": True, "result": tool_map[tool_name](params)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        return {"success": False, "error": f"未知工具: {tool_name}"}


if __name__ == "__main__":
    print("=== 工具测试 ===\n")

    print("1. 养老保险缴费计算（基数10000元）:")
    print(calculate_pension_contribution(10000))

    print("\n2. 医疗保险缴费计算（基数10000元，一档）:")
    print(calculate_medical_contribution(10000, "一档"))

    print("\n3. 五险一金全额计算:")
    result = calculate_all_insurances(10000, 10000, "一档", 10000, 0.12)
    for k, v in result.items():
        print(f"  {k}: {v}")
