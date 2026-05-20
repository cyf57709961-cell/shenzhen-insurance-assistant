# -*- coding: utf-8 -*-
"""
用户记忆数据类（Vercel 版 — 无文件持久化，画像由前端传入）
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class UserMemory:
    user_id: str = "default"
    city: str = "深圳"
    employment_status: str = ""
    household_type: str = ""
    pension_level: str = ""
    medical_level: str = ""
    contribution_base: float = 0
    housing_fund_base: float = 0
    housing_fund_rate: float = 0
    cumulative_years: float = 0
    monthly_salary: float = 0
    age: int = 0
    gender: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserMemory':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_summary(self) -> str:
        """生成用户画像摘要"""
        parts = []
        if self.employment_status:
            parts.append(f"就业状态: {self.employment_status}")
        if self.household_type:
            parts.append(f"户籍: {self.household_type}")
        if self.gender:
            parts.append(f"性别: {self.gender}")
        if self.age > 0:
            parts.append(f"年龄: {self.age}岁")
        if self.medical_level:
            parts.append(f"医保档次: {self.medical_level}")
        if self.monthly_salary > 0:
            parts.append(f"月工资: {self.monthly_salary}元")
        if self.cumulative_years > 0:
            parts.append(f"累计缴费: {self.cumulative_years}年")
        if self.housing_fund_rate > 0:
            parts.append(f"公积金比例: {int(self.housing_fund_rate * 100)}%")
        return " | ".join(parts) if parts else "暂无用户信息"
