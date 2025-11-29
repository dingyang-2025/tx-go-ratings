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
    st.subheader("🏆 实时排行 (Top Ratings)")

    # --- 1. 活跃筛选按钮 ---
    # 默认勾选，定义“活跃”为近 730 天（2年）
    active_only = st.checkbox("只看活跃 (近2年)", value=True)
    
    # --- 2. 计算统计数据 (总局数、胜率) ---
    stats = history_df.groupby('Name').agg(
        Total_Games=('Result', 'count'),                   
        Win_Count=('Result', lambda x: (x == 'Win').sum()) 
    )
    # 计算胜率
    stats['Win_Rate'] = (stats['Win_Count'] / stats['Total_Games'] * 100).round(1).astype(str) + '%'

    # --- 3. 准备基础数据 ---
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
        full_df = pd.merge(rank_df, stats, on='Name', how='left')
        full_df['Total_Games'] = full_df['Total_Games'].fillna(0).astype(int)
        full_df['Win_Rate'] = full_df['Win_Rate'].fillna('0.0%')

        # 【调整】：门槛改为 15 局
        threshold = 15
        display_df = full_df[full_df['Total_Games'] >= threshold].copy()

        # 活跃筛选
        if active_only:
            two_years_ago = pd.Timestamp.now() - pd.DateOffset(days=730)
            display_df['Last_Active'] = pd.to_datetime(display_df['Last_Active'])
            display_df = display_df[display_df['Last_Active'] >= two_years_ago]

        if not display_df.empty:
            # 排序：按分数降序
            display_df = display_df.sort_values(by='Rating', ascending=False).reset_index(drop=True)
            display_df.index += 1 

            # 整理列名
            display_df = display_df[['Name', 'Rating', 'Total_Games', 'Win_Rate']]
            display_df.columns = ['选手', '等级分', '总局数', '总胜率']
            
            # 使用 st.dataframe (可滚动)
            st.dataframe(display_df, use_container_width=True)
            
            # 底部动态文案
            st.caption(f"注：榜单仅显示总对局数 ≥ {threshold} 局的选手。")
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
            st.altair_chart(chart, width="stretch", height="content")
    else:
        st.info("暂无历史 Elo 数据（先录入几盘吧）。")

st.divider()


# ========== 选手详细档案 ==========
st.header("🔍 选手详细档案")

# --- 1. 选人交互 ---
all_players = sorted(ratings.keys())
selected_player = st.selectbox("选择选手查看详情:", ["(请选择)"] + all_players)

if selected_player != "(请选择)":
    # --- 2. 准备数据 ---
    curr_rating = int(ratings[selected_player])
    # 筛选出【个人】的历史记录
    player_history = history_df[history_df['Name'] == selected_player].copy()
    
    # 计算名次
    rank_text = "" 
    if 'display_df' in locals() and not display_df.empty:
        rank_search = display_df[display_df['选手'] == selected_player]
        if not rank_search.empty:
            r_val = rank_search.index[0]
            rank_text = f"第 {r_val} 名"
        else:
            rank_text = "未上榜"

    # 计算极值
    if not player_history.empty:
        max_rating = int(player_history['Rating'].max())
        min_rating = int(player_history['Rating'].min())
    else:
        max_rating = curr_rating
        min_rating = curr_rating

    # --- 3. 指标卡片 (维持经典布局) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前等级分", f"{curr_rating}", delta=rank_text if rank_text else None)
    with col2:
        st.metric("历史最高", f"{max_rating}")
    with col3:
        st.metric("历史最低", f"{min_rating}")

    # --- 4. 这里的布局严格还原截图 ---
    # 分隔线
    st.markdown("---")
    
    if not player_history.empty:
        # 预计算对手数据
        opp_stats = player_history.groupby('Opponent').agg(
            Games=('Result', 'count'),
            Wins=('Result', lambda x: (x == 'Win').sum())
        ).reset_index()
        opp_stats['Win_Rate'] = opp_stats['Wins'] / opp_stats['Games']

        # --- A. 🤝 老对手 (局数最多) ---
        # 规则：≥2局，按局数降序
        rivals = opp_stats[opp_stats['Games'] >= 2].sort_values(by='Games', ascending=False).head(5)

        # --- B. ☠️ 苦手 (胜率 < 50%) ---
        # 规则：≥2局，胜率<50%。排序：胜率升序(越惨越前) -> 局数降序
        nemesis = opp_stats[
            (opp_stats['Games'] >= 2) & 
            (opp_stats['Win_Rate'] < 0.5)
        ].sort_values(by=['Win_Rate', 'Games'], ascending=[True, False]).head(5)

        # --- C. 🍰 下手 (胜率 > 50%) ---
        # 规则：≥2局，胜率>50%。排序：胜率降序(越稳越前) -> 局数降序
        prey = opp_stats[
            (opp_stats['Games'] >= 2) & 
            (opp_stats['Win_Rate'] > 0.5)
        ].sort_values(by=['Win_Rate', 'Games'], ascending=[False, False]).head(5)

        # --- 经典三列布局 (纯文本列表) ---
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### 🤝 老对手") # 使用 Markdown 标题保持字号
            if not rivals.empty:
                for _, row in rivals.iterrows():
                    wins = row['Wins']
                    losses = row['Games'] - wins
                    st.write(f"{row['Opponent']} ({wins}胜{losses}负)")
            else:
                st.caption("暂无")

        with c2:
            st.markdown("### ☠️ 苦手") # 还原骷髅头图标
            if not nemesis.empty:
                for _, row in nemesis.iterrows():
                    wins = row['Wins']
                    losses = row['Games'] - wins
                    st.write(f"{row['Opponent']} ({wins}胜{losses}负)")
            else:
                st.caption("暂无")

        with c3:
            st.markdown("### 🍰 下手") # 还原蛋糕图标
            if not prey.empty:
                for _, row in prey.iterrows():
                    wins = row['Wins']
                    losses = row['Games'] - wins
                    st.write(f"{row['Opponent']} ({wins}胜{losses}负)")
            else:
                st.caption("暂无")

        # --- 5. 个人对局记录 (还原) ---
        st.subheader("📜 个人对局记录")
        # 按日期倒序
        ph_display = player_history.sort_values(by='Date', ascending=False).copy()
        #
