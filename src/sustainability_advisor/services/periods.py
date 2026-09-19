"""Reporting period helpers."""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, model_validator


class Period(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.end < self.start:
            raise ValueError("period end precedes period start")
        return self

    @property
    def months(self) -> int:
        """Number of calendar months touched by the period."""
        return (self.end.year - self.start.year) * 12 + self.end.month - self.start.month + 1


def month_start(value: date) -> date:
    return value.replace(day=1)


def add_months(value: date, months: int) -> date:
    index = value.year * 12 + (value.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def month_end(value: date) -> date:
    return add_months(month_start(value), 1) - timedelta(days=1)


def trailing_months(end: date, months: int) -> Period:
    """The ``months`` calendar months ending with the month containing ``end``."""
    last = month_start(end)
    return Period(start=add_months(last, -(months - 1)), end=last)
