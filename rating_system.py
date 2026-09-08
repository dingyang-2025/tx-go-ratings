"""腾讯围棋协会等级分计算规则。

v1 是项目原有规则：所有已登记对局等权，双方 K=32。
v2 使用动态 K 和赛事权重。等级分始终保留小数，只在页面显示时取整。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


INITIAL_RATING: Final[float] = 1500.0
RATING_SCALE: Final[float] = 400.0

VERSION_LABELS: Final[dict[str, str]] = {
    "v2": "v2 当前规则",
    "v1": "v1 旧规则（对照）",
}

STATUS_PROVISIONAL: Final[str] = "暂定"
STATUS_CALIBRATING: Final[str] = "校准中"
STATUS_STABLE: Final[str] = "稳定"

EVENT_WEIGHT_LABELS: Final[dict[float, str]] = {
    1.0: "现场赛",
    0.8: "预选赛",
    0.5: "协会杯赛",
    0.0: "未分类（不计分）",
}


@dataclass(frozen=True)
class RatingResult:
    ratings: dict[str, float]
    last_active: dict[str, pd.Timestamp]
    history: pd.DataFrame
    effective_games: dict[str, float]


HISTORY_COLUMNS: Final[list[str]] = [
    "Game_ID",
    "Date",
    "Name",
    "Rating_Before",
    "Rating",
    "Rating_Change",
    "Opponent",
    "Result",
    "Note1",
    "Note2",
    "Event_Weight",
    "Event_Type",
    "K_Factor",
    "Effective_Games_Before",
    "Effective_Games_After",
    "Rating_Status",
]


def standardize_name(name: object) -> str:
    """把 CSV 中的人名统一为无首尾空格的字符串。"""
    if name is None or pd.isna(name):
        return ""
    return str(name).strip()


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Elo 期望得分，保留原系统的 400 分尺度。"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / RATING_SCALE))


def classify_event(note1: object) -> tuple[str, float]:
    """根据赛事名称返回（赛事类型，权重）。

    第10届个人赛的历史数据只写了“A组”，实质上是预选赛，单独兼容。
    未识别的赛事保留在记录中，但不改变等级分，避免误把练习赛计入。
    """
    event_name = "" if note1 is None or pd.isna(note1) else str(note1).strip()

    if any(keyword in event_name for keyword in ("捉早杯", "捉早贺岁杯", "菜鸡杯")):
        return EVENT_WEIGHT_LABELS[0.5], 0.5
    if "现场赛" in event_name:
        return EVENT_WEIGHT_LABELS[1.0], 1.0
    if "预选赛" in event_name:
        return EVENT_WEIGHT_LABELS[0.8], 0.8
    if "第10届腾讯围棋大赛个人赛A组" in event_name:
        return EVENT_WEIGHT_LABELS[0.8], 0.8
    return EVENT_WEIGHT_LABELS[0.0], 0.0


def k_factor_for(effective_games_before: float) -> int:
    """v2 动态 K：前10个有效局48，10至30个40，之后28。"""
    if effective_games_before < 10:
        return 48
    if effective_games_before < 30:
        return 40
    return 28


def rating_status(effective_games: float) -> str:
    """把有效对局数翻译为用户可理解的分数状态。"""
    if effective_games < 10:
        return STATUS_PROVISIONAL
    if effective_games < 30:
        return STATUS_CALIBRATING
    return STATUS_STABLE


def _game_settings(version: str, note1: object, effective_games_before: float) -> tuple[str, float, int]:
    if version == "v1":
        return "旧版等权", 1.0, 32
    if version != "v2":
        raise ValueError(f"不支持的等级分版本：{version}")
    event_type, event_weight = classify_event(note1)
    return event_type, event_weight, k_factor_for(effective_games_before)


def calculate_ratings(
    df: pd.DataFrame,
    *,
    version: str = "v2",
    initial_rating: float = INITIAL_RATING,
) -> RatingResult:
    """按时间顺序重放对局并返回当前等级分、活跃时间和逐局历史。"""
    if version not in VERSION_LABELS:
        raise ValueError(f"不支持的等级分版本：{version}")
    if df is None or df.empty:
        return RatingResult({}, {}, pd.DataFrame(columns=HISTORY_COLUMNS), {})

    ratings: dict[str, float] = {}
    last_active: dict[str, pd.Timestamp] = {}
    effective_games: dict[str, float] = {}
    history: list[dict] = []

    # 同一天的多局比赛按 CSV 原始顺序计算，避免刷新后顺序随机变化。
    df_sorted = df.copy()
    df_sorted["__source_order"] = range(len(df_sorted))
    df_sorted = df_sorted.sort_values(["Date", "__source_order"], kind="mergesort")

    for game_id, row in df_sorted.iterrows():
        p1 = standardize_name(row.get("Player1"))
        p2 = standardize_name(row.get("Player2"))
        winner = standardize_name(row.get("Winner"))
        date = row.get("Date")

        if not p1 or not p2 or not winner or p1 == p2 or winner not in (p1, p2):
            continue

        for player in (p1, p2):
            ratings.setdefault(player, float(initial_rating))
            effective_games.setdefault(player, 0.0)
            last_active[player] = date

        rating_before = {p1: ratings[p1], p2: ratings[p2]}
        expected = {
            p1: calculate_expected_score(ratings[p1], ratings[p2]),
            p2: calculate_expected_score(ratings[p2], ratings[p1]),
        }
        actual = {p1: 1.0 if winner == p1 else 0.0, p2: 1.0 if winner == p2 else 0.0}

        note1 = row.get("Note1", "")
        note2 = row.get("Note2", "")
        # 赛事类型和权重与棋手无关；K 值按双方各自赛前状态计算。
        if version == "v1":
            event_type, event_weight = "旧版等权", 1.0
        else:
            event_type, event_weight = classify_event(note1)

        k_factors: dict[str, int] = {}
        for player in (p1, p2):
            _, _, k_factor = _game_settings(version, note1, effective_games[player])
            k_factors[player] = k_factor
            ratings[player] += k_factor * event_weight * (actual[player] - expected[player])

        loser = p2 if winner == p1 else p1
        for player in (p1, p2):
            effective_before = effective_games[player]
            effective_games[player] += event_weight
            history.append(
                {
                    "Game_ID": game_id,
                    "Date": date,
                    "Name": player,
                    "Rating_Before": rating_before[player],
                    "Rating": ratings[player],
                    "Rating_Change": ratings[player] - rating_before[player],
                    "Opponent": p2 if player == p1 else p1,
                    "Result": "Win" if player == winner else "Loss",
                    "Note1": note1,
                    "Note2": note2,
                    "Event_Weight": event_weight,
                    "Event_Type": event_type,
                    "K_Factor": k_factors[player],
                    "Effective_Games_Before": effective_before,
                    "Effective_Games_After": effective_games[player],
                    "Rating_Status": rating_status(effective_games[player]) if version == "v2" else "旧版",
                }
            )

    return RatingResult(
        ratings=ratings,
        last_active=last_active,
        history=pd.DataFrame(history, columns=HISTORY_COLUMNS),
        effective_games=effective_games,
    )
