import os
import datetime

import altair as alt
import pandas as pd
import streamlit as st

# ===============================
# 基础配置
# ===============================

# 数据文件路径：放在仓库根目录
BASE_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(BASE_DIR, "data.csv")

EXPECTED_COLUMNS = ["Date", "Player1", "Player2", "Winner", "Note1", "Note2"]


# ===============================
# 工具函数
# ===============================

def standardize_name(name: str) -> str:
    """人名统一处理：转成字符串、去掉首尾空格。"""
    if name is None:
        return ""
    return str(name).strip()


# --- Elo 相关 ---

def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_winner: float, rating_loser: float, k: int = 32) -> tuple[float, float]:
    expected_winner = calculate_expected_score(rating_winner, rating_loser)
    new_rating_winner = rating_winner + k * (1 - expected_winner)
    new_rating_loser = rating_loser + k * (0 - expected_winner)
    return new_rating_winner, new_rating_loser


# --- 数据加载 / 保存 ---

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """保证 df 至少包含 EXPECTED_COLUMNS 这些列，没有就补空字符串。"""
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # 多出来的列先保留在后面，方便以后扩展
    ordered = df[EXPECTED_COLUMNS + [c for c in df.columns if c not in EXPECTED_COLUMNS]]
    return ordered


def load_data() -> pd.DataFrame:
    """
    读取 data.csv：
    - 如果文件不存在，先创建空表；
    - 任何异常都返回一个结构正确但为空的 DataFrame，防止页面直接崩掉。
    """
    if not os.path.exists(FILE_PATH):
        empty = pd.DataFrame(columns=EXPECTED_COLUMNS)
        empty.to_csv(FILE_PATH, index=False)
        empty["Date"] = pd.to_datetime(empty.get("Date"))
        empty["Note"] = ""
        return empty

    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        # 读取失败时给个提示，但仍然保证页面可用
        st.error(f"读取数据文件失败：{e}")
        empty = pd.DataFrame(columns=EXPECTED_COLUMNS)
        empty["Date"] = pd.to_datetime(empty.get("Date"))
        empty["Note"] = ""
        return empty

    if df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = _ensure_columns(df)

    # 统一日期格式
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Note 列：Note1 | Note2
    df["Note1"] = df["Note1"].fillna("").astype(str)
    df["Note2"] = df["Note2"].fillna("").astype(str)
    df["Note"] = df["Note1"] + " | " + df["Note2"]

    return df


def save_game(date, p1, p2, winner, note1, note2) -> None:
    """往 data.csv 追加一行对局记录。"""
    p1_std = standardize_name(p1)
    p2_std = standardize_name(p2)
    winner_std = standardize_name(winner)

    if isinstance(date, (datetime.date, datetime.datetime)):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)

    new_row = pd.DataFrame(
        {
            "Date": [date_str],
            "Player1": [p1_std],
            "Player2": [p2_std],
            "Winner": [winner_std],
            "Note1": [note1 or ""],
            "Note2": [note2 or ""],
        }
    )

    header = not os.path.exists(FILE_PATH) or os.path.getsize(FILE_PATH) == 0
    # 直接以追加方式写入
    new_row.to_csv(FILE_PATH, mode="a", header=header, index=False)


# --- Elo 计算与历史 ---

def calculate_ratings(
    df: pd.DataFrame,
    initial_rating: int = 1500,
    k_factor: int = 32,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    根据对局记录计算：
    - ratings: {name -> rating}
    - last_active: {name -> 最近一局时间}
    - history_df: 每一局后的历史 Elo（给折线图 / 选手极值用）
    """
    history_columns = ["Date", "Name", "Rating", "Opponent", "Result", "Note1", "Note2"]

    if df is None or df.empty:
        return {}, {}, pd.DataFrame(columns=history_columns)

    ratings: dict[str, float] = {}
    last_active: dict[str, pd.Timestamp] = {}
    history: list[dict] = []

    # 先按日期排序，保证 Elo 时间顺序正确
    df_sorted = df.sort_values("Date")

    for _, row in df_sorted.iterrows():
        p1 = standardize_name(row.get("Player1"))
        p2 = standardize_name(row.get("Player2"))
        winner = standardize_name(row.get("Winner"))
        date = row.get("Date")

        # 数据不完整的直接跳过
        if not p1 or not p2 or not winner:
            continue
        if winner not in (p1, p2):
            # Winner 字段写错的对局也跳过，避免把 Elo 搞乱
            continue

        # 自动初始化等级分
        if p1 not in ratings:
            ratings[p1] = initial_rating
        if p2 not in ratings:
            ratings[p2] = initial_rating

        last_active[p1] = date
        last_active[p2] = date

        loser = p2 if winner == p1 else p1

        r_w = ratings[winner]
        r_l = ratings[loser]

        e_w = calculate_expected_score(r_w, r_l)
        e_l = 1 - e_w

        new_r_w = r_w + k_factor * (1 - e_w)
        new_r_l = r_l + k_factor * (0 - e_l)

        ratings[winner] = new_r_w
        ratings[loser] = new_r_l

        note1 = row.get("Note1", "")
        note2 = row.get("Note2", "")

        # 记录胜者
        history.append(
            {
                "Date": date,
                "Name": winner,
                "Rating": new_r_w,
                "Opponent": loser,
                "Result": "Win",
                "Note1": note1,
                "Note2": note2,
            }
        )
        # 记录负者
        history.append(
            {
                "Date": date,
                "Name": loser,
                "Rating": new_r_l,
                "Opponent": winner,
                "Result": "Loss",
                "Note1": note1,
                "Note2": note2,
            }
        )

    history_df = pd.DataFrame(history, columns=history_columns)
    return ratings, last_active, history_df


def get_rival_analysis(player_name: str, df: pd.DataFrame) -> list[dict]:
    """返回选手对手统计（总局数 / 胜率等）。"""
    if df is None or df.empty or not player_name:
        return []

    my_games = df[(df["Player1"] == player_name) | (df["Player2"] == player_name)]
    stats: dict[str, list[int]] = {}  # {opp: [wins, total]}

    for _, row in my_games.iterrows():
        if row["Player1"] == player_name:
            opp = row["Player2"]
        else:
            opp = row["Player1"]

        is_win = 1 if row["Winner"] == player_name else 0

        if opp not in stats:
            stats[opp] = [0, 0]
        stats[opp][0] += is_win
        stats[opp][1] += 1

    results: list[dict] = []
    for opp, (w, t) in stats.items():
        if not opp:
            continue
        results.append(
            {
                "name": opp,
                "total": t,
                "wins": w,
                "win_rate": (w / t) * 100,
            }
        )
    return results


# ===============================
# 页面主逻辑
# ===============================

st.set_page_config(page_title="公司围棋大脑", layout="wide")
st.title("Go Ratings & Stats 📊")

# --- 读取数据 & 计算 Elo ---
df = load_data()
ratings, last_active, history_df = calculate_ratings(df)

# 动态获选手名单（仅根据出现过的双方）
all_known_players = set(df["Player1"].dropna().unique()) | set(df["Player2"].dropna().unique())
known_names = sorted(n for n in all_known_players if n)


# ========== 侧边栏：录入新对局 ==========
with st.sidebar:
    st.header("📝 录入新对局")

    with st.form("add_game"):
        new_date = st.date_input("日期", value=datetime.date.today())

        p1 = st.selectbox(
            "选手1 (Player1)",
            ["(请选择)"] + known_names + ["(手动输入)"],
            index=0,
        )
        p2 = st.selectbox(
            "选手2 (Player2)",
            ["(请选择)"] + known_names + ["(手动输入)"],
            index=0,
        )

        if p1 == "(手动输入)":
            p1 = st.text_input("请输入选手1名字").strip()
        if p2 == "(手动输入)":
            p2 = st.text_input("请输入选手2名字").strip()

        winner_choice = st.radio("胜者", ["选手1胜", "选手2胜"], horizontal=True)

        note1 = st.text_input("赛事名称 (Note1)", placeholder="例如：12届腾赛")
        note2 = st.text_input("轮次 (Note2)", placeholder="例如：第一轮")

        submitted = st.form_submit_button("提交")

        if submitted:
            if not p1 or not p2 or p1 in ("(请选择)",) or p2 in ("(请选择)",) or p1 == p2:
                st.error("请完整填写选手信息，且两位选手不能相同。")
            else:
                final_winner = p1 if winner_choice == "选手1胜" else p2
                save_game(new_date, p1, p2, final_winner, note1, note2)
                st.success(f"已保存：{p1} vs {p2}（胜者：{final_winner}）")
                st.rerun()


# ========== 实时排行 & 多人 Elo 走势 ==========
col_rank, col_trend = st.columns([1, 2])

with col_rank:
    st.subheader("🏆 实时排行")
    active_only = st.checkbox("只看活跃（近2年）", value=True)

    if ratings:
        rank_rows = []
        now_ts = pd.Timestamp.now()
        for name, score in ratings.items():
            last_dt = last_active.get(name)
            if pd.isna(last_dt):
                continue
            if active_only and (now_ts - last_dt).days > 730:
                # 超过两年没下了
                continue
            rank_rows.append(
                {"选手": name, "分数": int(round(score)), "最后一局": last_dt.date()}
            )

        if rank_rows:
            rank_df = (
                pd.DataFrame(rank_rows)
                .sort_values("分数", ascending=False)
                .reset_index(drop=True)
            )
            rank_df.index += 1
            st.dataframe(rank_df, height=400, width="stretch")
        else:
            st.info("😴 暂无活跃选手")
    else:
        st.info("暂无任何对局记录")

with col_trend:
    st.subheader("📈 历史走势")
    if not history_df.empty and not ratings == {}:
        # 默认前 5 名
        top_players = [
            name
            for name, _ in sorted(
                ratings.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]
        selected = st.multiselect(
            "选择选手对比：",
            options=list(ratings.keys()),
            default=top_players,
        )
        if selected:
            cd = history_df[history_df["Name"].isin(selected)].copy()
            cd = cd.sort_values("Date")
            ymin, ymax = cd["Rating"].min() - 50, cd["Rating"].max() + 50
            chart = (
                alt.Chart(cd)
                .mark_line(point=True)
                .encode(
                    x="Date:T",
                    y=alt.Y("Rating:Q", scale=alt.Scale(domain=[ymin, ymax])),
                    color="Name:N",
                    tooltip=["Date:T", "Name:N", "Rating:Q"],
                )
                .interactive()
            )
            st.altair_chart(chart, width="stretch", height="content")
    else:
        st.info("暂无历史 Elo 数据（先录入几盘吧）。")

st.divider()


# ========== 选手详细档案 ==========
st.subheader("🔍 选手详细档案")
col_sel, col_stats = st.columns([1, 3])

if "current_selected_player" not in st.session_state:
    st.session_state.current_selected_player = "(请选择)"

with col_sel:
    target = st.selectbox(
        "选择选手查看详情：",
        ["(请选择)"] + sorted(list(ratings.keys())),
        key="current_selected_player",
    )

if target != "(请选择)":
    # 基础数据
    my_games = df[
        (df["Player1"] == target) | (df["Player2"] == target)
    ].sort_values("Date", ascending=False)
    total_games = len(my_games)
    wins = len(my_games[my_games["Winner"] == target])
    win_rate = (wins / total_games * 100) if total_games > 0 else 0.0
    curr_score = int(round(ratings.get(target, 1500)))

    # 历史 Elo 极值
    my_history = history_df[history_df["Name"] == target].sort_values("Date")
    if not my_history.empty:
        peak_row = my_history.loc[my_history["Rating"].idxmax()]
        low_row = my_history.loc[my_history["Rating"].idxmin()]
        peak_score = int(round(peak_row["Rating"]))
        low_score = int(round(low_row["Rating"]))
        peak_date = peak_row["Date"].strftime("%Y-%m-%d")
        low_date = low_row["Date"].strftime("%Y-%m-%d")
    else:
        peak_score = low_score = curr_score
        peak_date = low_date = "N/A"

    # 对手分析
    rival_data = get_rival_analysis(target, df)
    old_rivals = sorted(rival_data, key=lambda x: x["total"], reverse=True)[:3]
    nemesis = sorted(
        [r for r in rival_data if r["total"] >= 2],
        key=lambda x: x["win_rate"],
    )[:3]
    preys = sorted(
        [r for r in rival_data if r["total"] >= 2],
        key=lambda x: x["win_rate"],
        reverse=True,
    )[:3]

    with col_stats:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("当前等级分", curr_score)
        m2.metric("巅峰等级分", peak_score, delta=peak_date)
        m3.metric("最低等级分", low_score, delta=low_date, delta_color="inverse")
        m4.metric("总对局数", f"{total_games} 局")
        m5.metric("总胜率", f"{win_rate:.1f}%")

        st.divider()

        c_rival, c_nemesis, c_prey = st.columns(3)

        def format_list(data_list: list[dict]) -> str:
            if not data_list:
                return "无数据"
            lines = []
            for i, r in enumerate(data_list, start=1):
                lines.append(
                    f"**{i}. {r['name']}**（{r['total']}局，胜率 {r['win_rate']:.0f}%）"
                )
            return "\n\n".join(lines)

        with c_rival:
            st.markdown("#### 🤝 老对手（交手最多）")
            st.markdown(format_list(old_rivals))

        with c_nemesis:
            st.markdown("#### ☠️ 苦手（胜率最低）")
            st.caption("*(仅统计对局数 ≥ 2)*")
            st.markdown(format_list(nemesis))

        with c_prey:
            st.markdown("#### 🍲 下手（胜率最高）")
            st.caption("*(仅统计对局数 ≥ 2)*")
            st.markdown(format_list(preys))

    st.divider()

    # 个人完整对局记录
    st.markdown(f"#### 📜 {target} 完整对局记录")
    if not my_games.empty:
        display_games = my_games.rename(
            columns={
                "Date": "日期",
                "Player1": "选手1",
                "Player2": "选手2",
                "Winner": "获胜者",
                "Note": "备注",
            }
        ).copy()
        display_games["日期"] = pd.to_datetime(display_games["日期"]).dt.strftime(
            "%Y-%m-%d"
        )
        cols_to_show = ["日期", "选手1", "选手2", "获胜者", "备注"]
        st.dataframe(display_games[cols_to_show], width="stretch")
    else:
        st.info("暂无对局记录")

st.divider()

# ========== 全公司完整对局记录 ==========
st.subheader("📜 全公司完整对局记录")
if not df.empty:
    full_display = (
        df.sort_values("Date", ascending=False)
        .rename(
            columns={
                "Date": "日期",
                "Player1": "选手1",
                "Player2": "选手2",
                "Winner": "获胜者",
                "Note": "备注",
            }
        )
        .copy()
    )
    full_display["日期"] = pd.to_datetime(full_display["日期"]).dt.strftime(
        "%Y-%m-%d"
    )
    cols_to_show = ["日期", "选手1", "选手2", "获胜者", "备注"]
    st.dataframe(full_display[cols_to_show], width="stretch", height=500)
else:
    st.info("目前还没有任何对局记录。")
