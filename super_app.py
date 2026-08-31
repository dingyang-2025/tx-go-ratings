from __future__ import annotations
import os
import datetime
import requests
import altair as alt
import pandas as pd
import streamlit as st

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



def safe_dataframe(data, height=None):
    """Streamlit compat across old/new versions."""
    kwargs = {}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(data, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(data, **kwargs)

def safe_altair_chart(chart):
    """Altair chart compat across old/new versions."""
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart)

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

# --- 腾讯围棋抓取工具 ---
def fetch_txwq_content(chessid: str):
    """从腾讯接口获取 SGF 内容"""
    url = "http://happyapp.huanle.qq.com/cgi-bin/CommonMobileCGI/TXWQFetchChess"
    data = {"chessid": chessid}
    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        if js.get("result") == 0:
            return js.get("chess")
        else:
            st.error(f"API 报错: {js.get('resultstr')}")
            return None
    except Exception as e:
        st.error(f"连接失败: {e}")
        return None

# ===============================
# 页面主逻辑
# ===============================

st.set_page_config(page_title="公司围棋大脑", layout="wide")
st.title("Go Ratings & Stats 📊")

# --- 读取数据 & 计算 Elo ---
df = load_data()
ratings, last_active, history_df = calculate_ratings(df)

# 动态获选手名单（仅根据出现过的双方）
# 先用 standardize_name 清洗，再用中文拼音 + 英文在后的规则排序
p1_names = df["Player1"].dropna().map(standardize_name)
p2_names = df["Player2"].dropna().map(standardize_name)
all_known_players = set(p1_names) | set(p2_names)

# 去掉空名和 'nan' 之类异常
cleaned_players = [
    name
    for name in all_known_players
    if name and str(name).strip().lower() != "nan"
]

# 使用和其它地方一致的排序规则：中文按姓氏拼音，英文排在后面
known_names = sorted(cleaned_players, key=player_sort_key)


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
    
    st.divider()  # 加一条分割线
    
    # 新增：腾讯围棋抓取小工具
    st.header("🛠 实用工具")
    st.markdown("🔗 [转播大厅](https://go.7dm7va.top)")
    with st.expander("📥 腾讯围棋棋谱抓取"):
        st.caption("输入对局 ID 即可提取 SGF 文件")
        cid = st.text_input("Chess ID", placeholder="如: 1770092663030101341")
        if st.button("获取并准备下载"):
            if cid:
                with st.spinner("抓取中..."):
                    sgf_text = fetch_txwq_content(cid.strip())
                    if sgf_text:
                        st.success("抓取成功！")
                        # 提供下载按钮
                        st.download_button(
                            label="💾 点击下载 SGF",
                            data=sgf_text,
                            file_name=f"TXWQ_{cid}.sgf",
                            mime="text/plain"
                        )
            else:
                st.warning("请输入有效 ID")

# ========== 实时排行 & 多人 Elo 走势 ==========
col_rank, col_trend = st.columns([1, 2])

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
            '近三个月': 90,
            '近半年': 180,
            '近一年': 365,
        }
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
                'Rating': int(r),
                'Last_Active': last_active.get(p)
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

            # 只统计总局数 ≥ threshold 的选手
            threshold = 15
            display_df = full_df[full_df['Total_Games'] >= threshold].copy()

            # 活跃筛选：近 2 年
            if active_only:
                two_years_ago = pd.Timestamp.now() - pd.DateOffset(days=730)
                display_df['Last_Active'] = pd.to_datetime(display_df['Last_Active'])
                display_df = display_df[display_df['Last_Active'] >= two_years_ago]

            if not display_df.empty:
                # 使用我们自己的拼音排序 key 排
                display_df['Name_sorted'] = display_df['Name'].apply(player_sort_key)

                # 排序：先按等级分降序，再按拼音
                display_df = display_df.sort_values(
                    by=['Rating', 'Name_sorted'],
                    ascending=[False, True]
                ).reset_index(drop=True)
                display_df.index += 1

                # 处理勋章
                def decorate_name(row):
                    wins = int(row.get('Win_Count', 0) or 0)
                    badges = build_badges(row['Name'], wins)
                    if not badges:
                        return row['Name']
                    return f"{row['Name']}  {' · '.join(badges)}"

                display_df['Name'] = display_df.apply(decorate_name, axis=1)

                # 生成“近半年变化”列（↑ 12（8局）/ ↓ 8（3局）/ —）
                def format_change_cell(change, games):
                    if games <= 0 or pd.isna(change):
                        return '—'
                    try:
                        change = float(change)
                    except Exception:
                        return '—'
                    if change == 0:
                        return f"—（{games}局）"
                    arrow = '↑' if change > 0 else '↓'
                    return f"{arrow} {abs(int(change))}（{games}局）"

                display_df['Period_Change'] = display_df.apply(
                    lambda row: format_change_cell(row['Period_Change'], row['Period_Games']),
                    axis=1,
                )

                # 整理列名
                display_df = display_df[['Name', 'Rating', 'Period_Change', 'Total_Games', 'Win_Rate']]
                display_df.columns = ['选手', '等级分', f'{change_window_label}变化', '总局数', '总胜率']

                # 着色：涨分绿、跌分红
                def highlight_change(val):
                    if isinstance(val, str):
                        if val.startswith('↑'):
                            return 'color: #16a34a;'  # 绿色
                        if val.startswith('↓'):
                            return 'color: #dc2626;'  # 红色
                    return ''

                change_column = f'{change_window_label}变化'
                styled = display_df.style.map(highlight_change, subset=[change_column])
                safe_dataframe(styled)
                st.caption(
                    f"注：榜单仅显示总对局数 ≥ {threshold} 局的选手；"
                    f"变化统计为{change_window_label}内的等级分涨跌，括号内为该时段对局数。"
                )
            else:
                st.info(f"暂无满足条件的选手（需对局 ≥ {threshold} 且在活跃期内）。")
        else:
            st.info("暂无排名数据")


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
            safe_altair_chart(chart)
    else:
        st.info("暂无历史 Elo 数据（先录入几盘吧）。")

st.divider()


# ========== 选手详细档案 ==========
st.subheader("🔍 选手详细档案")
col_sel, col_stats = st.columns([1, 3])

if "current_selected_player" not in st.session_state:
    st.session_state.current_selected_player = "(请选择)"

with col_sel:
    # 使用自定义的按姓氏拼音排序，英文名排最后
    sorted_players = sorted(list(ratings.keys()), key=player_sort_key)

    target = st.selectbox(
        "选择选手查看详情：",
        ["(请选择)"] + sorted_players,
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
        # 5 个指标
        m1, m2, m3, m4, m5 = st.columns(5)

        # 在“当前等级分”下面加名次说明
        with m1:
            st.metric("当前等级分", curr_score)
            st.caption(rank_text)

        with m2:
            st.metric("巅峰等级分", peak_score, delta=peak_date)

        with m3:
            st.metric(
                "最低等级分",
                low_score,
                delta=low_date,
                delta_color="inverse",
            )

        with m4:
            st.metric("总对局数", f"{total_games} 局")

        with m5:
            st.metric("总胜率", f"{win_rate:.1f}%")

        # 荣誉徽章展示
        if player_badges:
            st.markdown(f"**荣誉标记：** {' · '.join(player_badges)}")
        else:
            st.caption("荣誉标记：暂无特殊称号")

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
            st.markdown("#### ☠️ 上手（胜率最低）")
            st.caption("*(仅统计对局数 ≥ 2，且胜率 < 50%)*")
            st.markdown(format_list(nemesis))

        with c_prey:
            st.markdown("#### 🍲 下手（胜率最高）")
            st.caption("*(仅统计对局数 ≥ 2，且胜率 > 50%)*")
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
        safe_dataframe(display_games[cols_to_show])
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
    safe_dataframe(full_display[cols_to_show], height=500)
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

    # 使用和选手档案相同的排序规则：中文按姓氏拼音，英文放最后
    all_players_sorted = sorted(cleaned_players, key=player_sort_key)
    player_options = ["(请选择)"] + all_players_sorted

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
            cols_to_show = ["日期", "选手1", "选手2", "获胜者", "备注"]
            safe_dataframe(display_h2h[cols_to_show], height=400)

# ========== 数据维护（最近 N 条记录） ==========
st.divider()
st.subheader("🛠 数据维护（最近对局记录）")

if df.empty:
    st.info("当前还没有任何对局记录。")
else:
    # 想只维护最近多少条，可以改这个数字
    N_RECENT = 10

    # 取最近 N 条对局（按日期倒序），保留原始索引，方便回写
    recent = df.sort_values("Date", ascending=False).head(N_RECENT).copy()
    recent = recent.reset_index().rename(columns={"index": "__row_id"})

    # 准备展示用的 DataFrame
    recent_display = recent[
        ["__row_id", "Date", "Player1", "Player2", "Winner", "Note1", "Note2"]
    ].copy()

    # 重命名成中文列名，便于看
    recent_display = recent_display.rename(
        columns={
            "Date": "日期",
            "Player1": "选手1",
            "Player2": "选手2",
            "Winner": "获胜者",
            "Note1": "备注1",
            "Note2": "备注2",
        }
    )

    # 增加一列“删除？”
    recent_display["删除?"] = False

    st.caption(f"仅展示最近 {len(recent_display)} 条对局，可在此修改字段或勾选删除。")
    if hasattr(st, "data_editor"):
        edited = st.data_editor(
            recent_display,
            num_rows="fixed",
            hide_index=True,
            key="data_maintain_editor",
        )
    else:
        st.warning("当前运行环境不支持在线编辑（data_editor），请升级 Streamlit 后使用。")
        edited = recent_display.copy()

    if st.button("💾 保存上述修改到 data.csv"):
        # 把中文列名映射回内部列名
        internal = edited.rename(
            columns={
                "日期": "Date",
                "选手1": "Player1",
                "选手2": "Player2",
                "获胜者": "Winner",
                "备注1": "Note1",
                "备注2": "Note2",
                "删除?": "__delete",
            }
        ).copy()

        # 遍历每一行，根据 __row_id 定位到原 df
        to_drop_indices = []
        for _, row in internal.iterrows():
            row_id = int(row["__row_id"])
            if row["__delete"]:
                to_drop_indices.append(row_id)
            else:
                # 更新原始 df 中对应行的内容
                df.loc[row_id, "Date"] = row["Date"]
                df.loc[row_id, "Player1"] = row["Player1"]
                df.loc[row_id, "Player2"] = row["Player2"]
                df.loc[row_id, "Winner"] = row["Winner"]
                df.loc[row_id, "Note1"] = row.get("Note1", "")
                df.loc[row_id, "Note2"] = row.get("Note2", "")

        # 统一删除需要删除的行
        if to_drop_indices:
            df = df.drop(index=to_drop_indices)

        # 重新生成合并后的 Note 列（保持和前面逻辑一致）
        df["Note1"] = df["Note1"].fillna("").astype(str)
        df["Note2"] = df["Note2"].fillna("").astype(str)
        df["Note"] = df["Note1"] + " | " + df["Note2"]

        # 覆盖写回 data.csv
        df.to_csv(FILE_PATH, index=False)

        st.success("已将修改写入 data.csv，页面将刷新以应用最新数据。")
        st.rerun()
