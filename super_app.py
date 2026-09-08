from __future__ import annotations
import os
import altair as alt
import pandas as pd
import streamlit as st

from rating_system import (
    STATUS_STABLE,
    VERSION_LABELS,
    calculate_ratings as calculate_rating_result,
    k_factor_for,
    rating_status,
)

# 可选：按中文拼音排序
try:
    from pypinyin import lazy_pinyin  # 需要在 requirements.txt 里加 pypinyin
except ImportError:
    lazy_pinyin = None


def player_sort_key(name: str):
    """
    选手排序规则：
    1. 中文名字在前，按姓氏拼音排序；
    2. 英文名字在后，按英文名排序。
    """
    if not name:
        return (0, "", "")

    name = str(name).strip()

    # 判断是否“英文名”（全是 ASCII 字符）
    is_english = all(ord(ch) < 128 for ch in name if not ch.isspace())

    if is_english:
        # 英文放在 group=1，最后；再按字母排序
        return (1, name.lower(), name)

    # 中文名：group=0，按姓氏拼音排
    if lazy_pinyin is not None:
        surname = name[0]
        try:
            py = lazy_pinyin(surname)[0].lower()
        except Exception:
            py = surname
    else:
        # 没装 pypinyin 时，退化为按汉字本身排序
        py = name

    return (0, py, name)



def safe_dataframe(data, height=None, **extra_kwargs):
    """Streamlit compat across old/new versions."""
    kwargs = dict(extra_kwargs)
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)

def safe_altair_chart(chart):
    """Altair chart compat across old/new versions."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

# ===============================
# 基础配置
# ===============================

# 数据文件路径：放在仓库根目录
BASE_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(BASE_DIR, "data.csv")

EXPECTED_COLUMNS = ["Date", "Player1", "Player2", "Winner", "Note1", "Note2"]

# ===============================
# 荣誉标记配置（你只要改这里就行）
# ===============================

# 历届个人赛冠军名单（示例：请按真实名单填充）
CHAMPION_PLAYERS: set[str] = {
    "刘博东",
    "彭天佐",
    "彭雄伟",
    "沈张毅",
    "薛亦涵",
    "赵东易",
    "黄博阳",
    "王行健",
    # ...
}

# “百胜”门槛
WIN_MILESTONE = 100


def build_badges(name: str, wins: int | None = None) -> list[str]:
    """
    根据名字 + 胜局数，返回要展示的徽章列表：
    - 👑 腾冠：历届个人赛冠军
    - 💯 百胜：胜局数 >= WIN_MILESTONE
    """
    badges: list[str] = []
    if name in CHAMPION_PLAYERS:
        badges.append("👑")
    if wins is not None and wins >= WIN_MILESTONE:
        badges.append("💯")
    return badges


# ===============================
# 工具函数
# ===============================

def standardize_name(name: str) -> str:
    """人名统一处理：转成字符串、去掉首尾空格。"""
    if name is None:
        return ""
    return str(name).strip()


def count_recent_games(df: pd.DataFrame, start_date: pd.Timestamp) -> dict[str, int]:
    """统计每位选手从指定日期起至今天参加的实际对局数。"""
    if df is None or df.empty:
        return {}

    today = pd.Timestamp.now().normalize()
    dates = pd.to_datetime(df["Date"], errors="coerce")
    recent = df[(dates >= start_date) & (dates <= today)]
    names = pd.concat([recent["Player1"], recent["Player2"]], ignore_index=True)
    names = names.dropna().map(standardize_name)
    names = names[(names != "") & (names.str.lower() != "nan")]
    return {name: int(count) for name, count in names.value_counts().items()}


def recent_activity_sort_key(name: str, game_counts: dict[str, int]):
    """近一年对局数多者优先；局数相同时沿用拼音排序。"""
    return (-int(game_counts.get(name, 0)), *player_sort_key(name))


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

st.set_page_config(
    page_title="公司围棋大脑",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding: 0.8rem 0.65rem 2.5rem;
        }
        h1 { font-size: 1.85rem !important; }
        h2, h3 { line-height: 1.25 !important; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stDataFrame"] { font-size: 0.86rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Go Ratings & Stats 📊")

# --- 读取数据 & 选择等级分规则 ---
df = load_data()
one_year_ago = pd.Timestamp.now().normalize() - pd.DateOffset(years=1)
recent_game_counts = count_recent_games(df, one_year_ago)

with st.expander("⚙️ 等级分规则与版本", expanded=False):
    rating_version_label = st.radio(
        "计算版本",
        options=list(VERSION_LABELS.values()),
        index=0,
        horizontal=True,
        help="默认使用 v2；v1 仅用于查看规则升级前的结果。",
    )
    rating_version = next(
        version for version, label in VERSION_LABELS.items() if label == rating_version_label
    )
    if rating_version == "v2":
        st.markdown(
            """
            - **动态 K：** 有效对局少于 10 局用 48，10～不足 30 局用 40，30 局起用 28。
            - **赛事权重：** 现场赛 1.0、预选赛 0.8、捉早杯/贺岁杯/菜鸡杯 0.5。
            - **双方独立：** 每位棋手使用自己的 K 值，成熟棋手不会因遇到新人而一起剧烈波动。
            """
        )
    else:
        st.markdown(
            """
            - **统一初始分：** 所有选手都从 1500 分开始。
            - **固定 K：** 不区分新老选手，双方每局都使用 K=32。
            - **赛事等权：** 所有已登记对局的权重均为 1.0，不区分现场赛、预选赛或杯赛。
            - **用途：** 仅用于查看规则升级前的旧榜单，不作为当前默认结果。
            """
        )
        snapshot_path = os.path.join(BASE_DIR, "snapshots", "rating_v1_2026-09-08.csv")
        if os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as snapshot_file:
                st.download_button(
                    "下载 v1 迁移快照",
                    data=snapshot_file.read(),
                    file_name=os.path.basename(snapshot_path),
                    mime="text/csv",
                )

rating_result = calculate_rating_result(df, version=rating_version)
ratings = rating_result.ratings
last_active = rating_result.last_active
history_df = rating_result.history
effective_games = rating_result.effective_games
st.caption(
    "当前采用：**动态 K + 赛事权重（v2）**。需要时可展开上方规则切换旧版对照。"
    if rating_version == "v2"
    else "当前正在查看：**v1 旧规则**。这是对照视图，不是默认榜单。"
)

# ========== 实时排行 & 多人 Elo 走势 ==========
# 榜单和走势图上下排列并各占整行，避免任何窗口宽度下互相挤压。
col_rank = st.container()
col_trend = st.container()

with col_rank:
    st.subheader("🏆 实时排行 (Top Ratings)")

    # --- 1. 活跃筛选按钮 ---
    # 默认勾选，定义“活跃”为近 730 天（2年）
    active_only = st.checkbox("只看活跃 (近2年)", value=True)

    if history_df.empty or not ratings:
        st.info("暂无排名数据")
    else:
        # --- 2. 计算统计数据 (总局数、胜率) ---
        stats = history_df.groupby('Name').agg(
            Total_Games=('Result', 'count'),
            Win_Count=('Result', lambda x: (x == 'Win').sum())
        )
        stats['Win_Rate'] = (stats['Win_Count'] / stats['Total_Games'] * 100).round(1).astype(str) + '%'

        # --- 2.1 计算一段时间内的等级分变化与对局数 ---
        # “上一局涨跌”会在很久不下棋后仍然显示，容易被误读成近况。
        # 因此这里按选定时间段，计算期末等级分相对期初的变化。
        change_window_days = {
            '近半年': 180,
            '近一年': 365,
        }
        if (
            'rank_change_window' in st.session_state
            and st.session_state['rank_change_window'] not in change_window_days
        ):
            del st.session_state['rank_change_window']
        change_window_label = st.selectbox(
            '变化周期',
            options=list(change_window_days),
            index=1,
            key='rank_change_window',
        )
        period_days = change_window_days[change_window_label]
        period_start = pd.Timestamp.now().normalize() - pd.DateOffset(days=period_days)
        h_sorted = history_df.sort_values(['Name', 'Date']).copy()

        # 每人这段时间实际下了几局；没有下棋的人不显示陈旧的涨跌。
        period_games = (
            h_sorted[h_sorted['Date'] >= period_start]
            .groupby('Name')
            .size()
            .rename('Period_Games')
            .reset_index()
        )

        # 找到时间段开始前的最后一个等级分，作为比较基准。
        # 新选手在时间段内首次出现时，以初始分 1500 为基准。
        rating_before_period = (
            h_sorted[h_sorted['Date'] < period_start]
            .groupby('Name')
            .tail(1)[['Name', 'Rating']]
            .rename(columns={'Rating': 'Rating_Before_Period'})
        )

        # --- 3. 组装当前等级分 & 最近活跃时间 ---
        rank_data = []
        for p, r in ratings.items():
            rank_data.append({
                'Name': p,
                'Rating': float(r),
                'Last_Active': last_active.get(p),
                'Effective_Games': float(effective_games.get(p, 0)),
                'Rating_Status': rating_status(effective_games.get(p, 0)) if rating_version == 'v2' else '旧版',
            })
        rank_df = pd.DataFrame(rank_data)

        # --- 4. 合并与多重筛选 ---
        if not rank_df.empty:
            full_df = (rank_df
                       .merge(stats, on='Name', how='left')
                       .merge(period_games, on='Name', how='left')
                       .merge(rating_before_period, on='Name', how='left'))
            full_df['Total_Games'] = full_df['Total_Games'].fillna(0).astype(int)
            full_df['Win_Rate'] = full_df['Win_Rate'].fillna('0.0%')
            full_df['Win_Count'] = full_df['Win_Count'].fillna(0).astype(int)
            full_df['Period_Games'] = full_df['Period_Games'].fillna(0).astype(int)
            full_df['Rating_Before_Period'] = full_df['Rating_Before_Period'].fillna(1500)
            full_df['Period_Change'] = (
                full_df['Rating'] - full_df['Rating_Before_Period']
            ).where(full_df['Period_Games'] > 0)
            # Streamlit 会把空值显示成 None。用 0 作为“本周期未对局”的
            # 内部占位，展示时再格式化为“—”；对局列仍可明确区分两种情况。
            full_df['Period_Change'] = full_df['Period_Change'].fillna(0)

            # 只统计总局数 ≥ threshold 的选手
            threshold = 15
            display_df = full_df[full_df['Total_Games'] >= threshold].copy()

            # 活跃筛选：近 2 年
            if active_only:
                two_years_ago = pd.Timestamp.now() - pd.DateOffset(days=730)
                display_df['Last_Active'] = pd.to_datetime(display_df['Last_Active'])
                display_df = display_df[display_df['Last_Active'] >= two_years_ago]

            if not display_df.empty:
                display_df['Name_sorted'] = display_df['Name'].apply(player_sort_key)
                display_df = display_df.sort_values(
                    by=['Rating', 'Name_sorted'],
                    ascending=[False, True]
                ).reset_index(drop=True)
                display_df['Rank'] = range(1, len(display_df) + 1)
                display_df['_Sortable_Change'] = display_df['Period_Change'].where(
                    display_df['Period_Games'] > 0
                )

                view_mode = st.radio(
                    "榜单视图",
                    ["简洁榜单", "完整表格"],
                    horizontal=True,
                    key="rank_view_mode",
                )
                sort_mode = st.selectbox(
                    "排序依据",
                    ["等级分（高到低）", "近期涨分最多", "近期跌分最多", "近期对局最多"],
                    key="rank_sort_mode",
                )

                sort_rules = {
                    "等级分（高到低）": (['Rating', 'Name_sorted'], [False, True]),
                    "近期涨分最多": (['_Sortable_Change', 'Rating'], [False, False]),
                    "近期跌分最多": (['_Sortable_Change', 'Rating'], [True, False]),
                    "近期对局最多": (['Period_Games', 'Rating'], [False, False]),
                }
                sort_columns, sort_ascending = sort_rules[sort_mode]
                display_df = display_df.sort_values(
                    sort_columns,
                    ascending=sort_ascending,
                    na_position='last',
                )

                # 处理勋章
                def decorate_name(row):
                    wins = int(row.get('Win_Count', 0) or 0)
                    badges = build_badges(row['Name'], wins)
                    suffixes = list(badges)
                    if rating_version == 'v2' and row['Rating_Status'] != STATUS_STABLE:
                        suffixes.append(row['Rating_Status'])
                    if not suffixes:
                        return row['Name']
                    return f"{row['Name']}  {' · '.join(suffixes)}"

                display_df['Name'] = display_df.apply(decorate_name, axis=1)

                table_df = display_df[
                    ['Rank', 'Name', 'Rating', 'Period_Change', 'Period_Games', 'Total_Games', 'Win_Rate']
                ].copy()
                table_df.columns = [
                    '排名', '选手', '等级分', '变化', '对局', '总局数', '总胜率'
                ]
                table_df = table_df.set_index('排名')

                if view_mode == "简洁榜单":
                    table_df = table_df[['选手', '等级分', '变化', '对局']]
                else:
                    table_df = table_df[
                        ['选手', '等级分', '变化', '对局', '总局数', '总胜率']
                    ]

                change_column = '变化'

                def format_change_cell(change):
                    if pd.isna(change) or float(change) == 0:
                        return '—'
                    change = float(change)
                    arrow = '↑' if change > 0 else '↓'
                    return f"{arrow} {abs(int(round(change)))}"

                def highlight_change(val):
                    if pd.notna(val):
                        if val > 0:
                            return 'color: #16a34a;'
                        if val < 0:
                            return 'color: #dc2626;'
                    return ''

                styled = (
                    table_df.style
                    .map(highlight_change, subset=[change_column])
                    .format({'等级分': '{:.0f}', change_column: format_change_cell}, na_rep='—')
                )
                safe_dataframe(styled)
                st.caption(
                    f"榜单显示总对局数 ≥ {threshold} 局的选手；"
                    f"变化和对局均统计{change_window_label}。"
                )
                if rating_version == 'v2':
                    st.caption("姓名后的“暂定 / 校准中”表示有效对局不足 30；没有标注的为稳定状态。")
            else:
                st.info(f"暂无满足条件的选手（需对局 ≥ {threshold} 且在活跃期内）。")
        else:
            st.info("暂无排名数据")


with col_trend:
    st.divider()
    st.subheader("📈 历史走势")
    if not history_df.empty and not ratings == {}:
        # 手机端默认只展示 3 人，减少标签和图例拥挤；仍可手动增加。
        top_players = [
            name
            for name, _ in sorted(
                ratings.items(), key=lambda x: x[1], reverse=True
            )[:3]
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
            safe_altair_chart(chart)
    else:
        st.info("暂无历史 Elo 数据（先录入几盘吧）。")

st.divider()


# ========== 选手详细档案 ==========
st.subheader("🔍 选手详细档案")
col_sel = st.container()
col_stats = st.container()

if "current_selected_player" not in st.session_state:
    st.session_state.current_selected_player = "(请选择)"

with col_sel:
    # 近期更活跃的选手优先；活跃度相同时再按拼音排序。
    sorted_players = sorted(
        list(ratings.keys()),
        key=lambda name: recent_activity_sort_key(name, recent_game_counts),
    )

    target = st.selectbox(
        "选择选手查看详情：",
        ["(请选择)"] + sorted_players,
        key="current_selected_player",
    )
    st.caption("按近1年对局数从多到少排列；对局数相同时按拼音排列。")

if target != "(请选择)":
    # 基础数据
    my_games = df[
        (df["Player1"] == target) | (df["Player2"] == target)
    ].sort_values("Date", ascending=False)
    total_games = len(my_games)
    wins = len(my_games[my_games["Winner"] == target])
    win_rate = (wins / total_games * 100) if total_games > 0 else 0.0
    curr_score = int(round(ratings.get(target, 1500)))
    effective_count = float(effective_games.get(target, 0))
    current_status = rating_status(effective_count) if rating_version == "v2" else "旧版"
    next_k = k_factor_for(effective_count) if rating_version == "v2" else 32

    # 当前选手的荣誉徽章
    player_badges = build_badges(target, wins)

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

    # ===== 1）计算名次：在总对局 ≥ 15 局选手中的等级分排名 =====
    rank_text = "名次：—"
    threshold_rank = 15
    if not history_df.empty:
        # 每个选手的总局数
        stats_by_player = history_df.groupby("Name").agg(
            Total_Games=("Result", "count")
        )
        total_games_dict = stats_by_player["Total_Games"].to_dict()

        # 只保留总局数 ≥ threshold_rank 的选手
        ranking_list = []
        for name, rating in ratings.items():
            tg = int(total_games_dict.get(name, 0))
            if tg >= threshold_rank:
                ranking_list.append(
                    {
                        "Name": name,
                        "Rating": int(round(rating)),
                        "Total_Games": tg,
                    }
                )

        total_qualified = len(ranking_list)
        if total_qualified > 0:
            ranking_list_sorted = sorted(
                ranking_list, key=lambda x: x["Rating"], reverse=True
            )
            rank = None
            for idx, row in enumerate(ranking_list_sorted, start=1):
                if row["Name"] == target:
                    rank = idx
                    break

            if rank is not None:
                rank_text = f"名次：第 {rank} / 共 {total_qualified} 人（≥{threshold_rank} 局）"
            else:
                rank_text = f"名次：未上榜（对局数 < {threshold_rank} 局）"
    else:
        rank_text = "名次：暂无数据"

    # 对手分析
    rival_data = get_rival_analysis(target, df)

    # ===== 2）老对手、上手、下手规则 =====
    TOP_N = 5

    # 老对手：按总局数降序，取前 5 个
    old_rivals = sorted(
        rival_data, key=lambda x: x["total"], reverse=True
    )[:TOP_N]

    # 上手：总局数 ≥ 2 且胜率 < 50%，按「胜率升序，再按局数降序」排序
    nemesis_candidates = [
        r
        for r in rival_data
        if r["total"] >= 2 and r["win_rate"] < 50
    ]
    nemesis = sorted(
        nemesis_candidates,
        key=lambda x: (x["win_rate"], -x["total"]),
    )[:TOP_N]

    # 下手：总局数 ≥ 2 且胜率 > 50%，按「胜率降序，再按局数降序」排序
    preys_candidates = [
        r
        for r in rival_data
        if r["total"] >= 2 and r["win_rate"] > 50
    ]
    preys = sorted(
        preys_candidates,
        key=lambda x: (-x["win_rate"], -x["total"]),
    )[:TOP_N]

    with col_stats:
        # 每行最多两个指标，手机上仍能清楚阅读。
        m1, m2 = st.columns(2)

        # 在“当前等级分”下面加名次说明
        with m1:
            st.metric("当前等级分", curr_score)
            st.caption(rank_text)

        with m2:
            st.metric("分数状态", current_status)
            st.caption(f"下一局个人 K 值：{next_k}")

        m3, m4 = st.columns(2)

        with m3:
            effective_text = f"{effective_count:.1f}".rstrip("0").rstrip(".")
            st.metric("有效对局", f"{effective_text} 局")
            st.caption("已按赛事重要性折算")

        with m4:
            st.metric("总对局数", f"{total_games} 局")

        m5, m6 = st.columns(2)

        with m5:
            st.metric("巅峰等级分", peak_score, delta=peak_date)

        with m6:
            st.metric(
                "最低等级分",
                low_score,
                delta=low_date,
                delta_color="inverse",
            )

        st.metric("总胜率", f"{win_rate:.1f}%")

        # 荣誉徽章展示
        if player_badges:
            st.markdown(f"**荣誉标记：** {' · '.join(player_badges)}")
        else:
            st.caption("荣誉标记：暂无特殊称号")

        st.divider()

        c_rival, c_nemesis, c_prey = st.tabs(["🤝 老对手", "☠️ 上手", "🍲 下手"])

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
            st.caption("交手次数最多的 5 位对手")
            st.markdown(format_list(old_rivals))

        with c_nemesis:
            st.caption("仅统计交手 ≥ 2 局且胜率低于 50% 的对手")
            st.markdown(format_list(nemesis))

        with c_prey:
            st.caption("仅统计交手 ≥ 2 局且胜率高于 50% 的对手")
            st.markdown(format_list(preys))

    st.divider()

    # 个人完整对局记录
    st.markdown(f"#### 📜 {target} 完整对局记录")
    if not my_games.empty:
        # 详情只保留个人视角所需信息，避免重复显示选手1/选手2/获胜者。
        display_games = my_history.sort_values(
            ["Date", "Game_ID"], ascending=[False, False]
        ).copy()
        display_games["日期"] = pd.to_datetime(display_games["Date"]).dt.strftime("%Y-%m-%d")
        display_games["对手"] = display_games["Opponent"]
        display_games["赛果"] = display_games["Result"].map({"Win": "胜", "Loss": "负"})
        display_games["等级分变化"] = display_games["Rating_Change"]
        display_games["赛后分"] = display_games["Rating"]
        display_games["计分参数"] = display_games.apply(
            lambda row: f"K{int(row['K_Factor'])} × {row['Event_Weight']:.1f}", axis=1
        )
        display_games["赛事"] = display_games.apply(
            lambda row: " · ".join(
                part for part in (str(row.get("Note1", "")).strip(), str(row.get("Note2", "")).strip())
                if part and part.lower() != "nan"
            ),
            axis=1,
        )
        games_view = st.radio(
            "对局记录视图",
            ["简洁记录", "计分详情"],
            horizontal=True,
            key="player_games_view",
        )
        if games_view == "简洁记录":
            cols_to_show = ["日期", "对手", "赛果", "等级分变化"]
        else:
            cols_to_show = ["日期", "对手", "赛果", "等级分变化", "赛后分", "计分参数", "赛事"]

        def format_rating_change(change):
            if pd.isna(change) or abs(change) < 0.05:
                return "—"
            arrow = "↑" if change > 0 else "↓"
            sign = "+" if change > 0 else "−"
            return f"{arrow} {sign}{abs(change):.1f}"

        def highlight_rating_change(change):
            if pd.notna(change):
                if change > 0:
                    return "color: #16a34a;"
                if change < 0:
                    return "color: #dc2626;"
            return ""

        styled_games = (
            display_games[cols_to_show].style
            .map(highlight_rating_change, subset=["等级分变化"])
            .format(
                {"等级分变化": format_rating_change, "赛后分": "{:.0f}"},
                na_rep="—",
            )
        )
        safe_dataframe(styled_games, hide_index=True)
    else:
        st.info("暂无对局记录")

st.divider()

# ========== 全公司完整对局记录 ==========
with st.expander("📜 全公司完整对局记录", expanded=False):
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
        safe_dataframe(full_display[cols_to_show], height=500, hide_index=True)
    else:
        st.info("目前还没有任何对局记录。")

# ========== 查询交手记录 ==========
st.divider()
st.subheader("🤝 查询交手记录")

if df.empty:
    st.info("目前还没有任何对局记录，无法查询交手情况。")
else:
    # 提取所有出现过的选手姓名，先用 standardize_name 清理，再按拼音排序
    p1_names = df["Player1"].dropna().map(standardize_name)
    p2_names = df["Player2"].dropna().map(standardize_name)
    all_players_set = set(p1_names) | set(p2_names)

    # 去掉空字符串和 'nan' 之类的异常
    cleaned_players = [
        name
        for name in all_players_set
        if name and str(name).strip().lower() != "nan"
    ]

    # 使用和选手档案相同的排序规则：近期活跃度优先，其次按拼音。
    all_players_sorted = sorted(
        cleaned_players,
        key=lambda name: recent_activity_sort_key(name, recent_game_counts),
    )
    player_options = ["(请选择)"] + all_players_sorted

    st.caption("按近1年对局数从多到少排列；对局数相同时按拼音排列。")

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("选手 A", player_options, key="h2h_player_a")
    with col_b:
        player_b = st.selectbox("选手 B", player_options, key="h2h_player_b")

    if player_a == "(请选择)" or player_b == "(请选择)":
        st.info("请选择两个选手以查询交手记录。")
    elif player_a == player_b:
        st.warning("请不要选择同一个选手。")
    else:
        # 过滤两人之间的全部对局（双向匹配）
        mask = (
            ((df["Player1"] == player_a) & (df["Player2"] == player_b))
            | ((df["Player1"] == player_b) & (df["Player2"] == player_a))
        )
        h2h_games = df[mask].sort_values("Date", ascending=False)

        total_h2h = len(h2h_games)
        if total_h2h == 0:
            st.info(f"目前没有 {player_a} 与 {player_b} 的对局记录。")
        else:
            wins_a = (h2h_games["Winner"] == player_a).sum()
            wins_b = (h2h_games["Winner"] == player_b).sum()
            others = total_h2h - wins_a - wins_b

            col_total, col_a_stat, col_b_stat = st.columns(3)
            with col_total:
                st.metric("交手总局数", f"{total_h2h} 局")
            with col_a_stat:
                st.metric(f"{player_a} 胜局数", f"{wins_a} 局")
            with col_b_stat:
                st.metric(f"{player_b} 胜局数", f"{wins_b} 局")

            if others > 0:
                st.caption(f"其中有 {others} 局未能判定胜负（或记录异常）。")

            st.markdown(f"##### 📜 {player_a} vs {player_b} 具体对局记录")

            display_h2h = (
                h2h_games.rename(
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
            display_h2h["日期"] = pd.to_datetime(display_h2h["日期"]).dt.strftime(
                "%Y-%m-%d"
            )
            # 两位选手已在上方明确，手机端只需显示日期、胜者和赛事。
            cols_to_show = ["日期", "获胜者", "备注"]
            safe_dataframe(display_h2h[cols_to_show], height=400, hide_index=True)
