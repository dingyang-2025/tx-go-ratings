import os
import json
import time
import datetime
import requests
import base64  # 👈 新增：用于撕开乱码包装
import zlib    # 👈 新增：用于解压数据
import pandas as pd  # 👈 补全了！修复 NameError
import altair as alt 
import streamlit as st
from urllib.parse import urlparse, parse_qs

# Selenium 核心库
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 辅助函数：数字坐标转 SGF 字母
def num_to_sgf(n):
    return chr(ord('a') + n)

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

# --- 腾讯围棋抓取工具 (终极修正：精准窃听版) ---
def find_moves_in_dict(obj):
    """
    在截获的 chessData 对象中，自动寻找最像棋谱的数组
    """
    if isinstance(obj, list):
        # 特征：列表长度>10，且第一个元素也是列表 [x, y, z]
        if len(obj) > 10 and isinstance(obj[0], list) and len(obj[0]) >= 3:
            # 验证元素是否为数字
            try:
                if isinstance(obj[0][0], (int, float)): return obj
            except: pass
        for item in obj:
            res = find_moves_in_dict(item)
            if res: return res
    elif isinstance(obj, dict):
        for value in obj.values():
            res = find_moves_in_dict(value)
            if res: return res
    return None

def fetch_txwq_websocket(input_str: str):
    """
    利用 Protobuf 解密后的 Console Log 进行截获 (自动过滤 undefined 包)
    """
    input_str = input_str.strip()
    full_share_url = input_str
    if "txwqshare" in input_str or "h5.txwq.qq.com" in input_str:
        pass
    else:
        return None, "⚠️ 请输入完整的直播分享链接。"

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)

        # 👑 核心：注入“挑食”的窃听代码
        hijack_script = """
        window.__captured_chess_data = []; // 用数组存，防止漏掉
        
        var originalLog = console.log;
        var originalInfo = console.info;
        
        function checkAndCapture(args) {
            for (var i = 0; i < args.length; i++) {
                var arg = args[i];
                if (arg && typeof arg === 'object') {
                    // 🛡️ 过滤器：只有当 chessData 存在且不是 undefined 时才抓！
                    // 这完美避开了你发现的 type: 402 空包
                    if (arg.chessData !== undefined && arg.chessData !== null) {
                        window.__captured_chess_data.push(arg.chessData);
                    }
                    // 双保险：有时候它直接把数据叫 moves 或 subChunks
                    else if (arg.moves || arg.subChunks) {
                         window.__captured_chess_data.push(arg);
                    }
                }
            }
        }

        console.log = function() { checkAndCapture(arguments); originalLog.apply(console, arguments); };
        console.info = function() { checkAndCapture(arguments); originalInfo.apply(console, arguments); };
        """
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': hijack_script})

        st.toast("控制台窃听器已植入，正在等待有效数据...")
        driver.get(full_share_url)
        
        raw_moves = None
        captured_list = []
        
        # 轮询 12 秒，确保抓到那个 type: 203 的包
        for i in range(12): 
            time.sleep(1)
            captured_list = driver.execute_script("return window.__captured_chess_data;")
            
            if captured_list and len(captured_list) > 0:
                # 遍历抓到的所有非空包
                for data_item in captured_list:
                    # 在每个包里找棋谱
                    found = find_moves_in_dict(data_item)
                    if found:
                        raw_moves = found
                        break
            if raw_moves:
                st.toast(f"🎉 成功在第 {i+1} 秒捕获并解析棋谱！")
                break
        
        if not raw_moves:
             debug_len = len(captured_list) if captured_list else 0
             return None, f"❌ 监听超时。捕获了 {debug_len} 个有效包，但未识别出棋谱格式。\n(已自动忽略了 type: 402 这种空包)"

        # 组装 SGF
        sgf_header = f"(;GM[1]SZ[19]AP[Txwq_Console_Hack]DT[{datetime.date.today()}]"
        sgf_moves = ""
        move_count = 0
        for move in raw_moves:
            try:
                # 腾讯 Protobuf 解密后的格式：[x, y, color] 或 [color, x, y]
                # 过滤出所有 0-18 的数字作为坐标
                nums = [x for x in move if isinstance(x, (int, float))]
                
                # 启发式规则：找最后两个 <=18 的数字当坐标
                coords = [n for n in nums if 0 <= n <= 18]
                if len(coords) >= 2:
                    x, y = int(coords[-2]), int(coords[-1])
                    
                    # 颜色判断：如果有明确的 0/1 就用，没有就按步数黑白交替
                    c = "B" if move_count % 2 == 0 else "W"
                    # 尝试寻找颜色标识 (通常是第一个数)
                    if nums[0] == 0: c = "B"
                    elif nums[0] == 1: c = "W"

                    sgf_moves += f";{c}[{num_to_sgf(x)}{num_to_sgf(y)}]"
                    move_count += 1
            except: continue

        return sgf_header + sgf_moves + ")", f"✅ 完美破局！过滤掉空包后，成功提取 {move_count} 手。"

    except Exception as e:
        return None, f"❌ 运行异常: {str(e)}"
    finally:
        if driver: driver.quit()

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
    
    st.divider()
    st.header("🛠 实用工具")
    with st.expander("📡 腾讯围棋直播抓取 (解码版)", expanded=True):
        st.caption("自动识别 Base64/Zlib 加密数据流")
        cid = st.text_input("输入直播分享链接", placeholder="https://h5.txwq.qq.com/txwqshare/...")
        
        if st.button("开始监听"):
            if cid:
                with st.spinner("正在解码直播数据流..."):
                    sgf_text, status_msg = fetch_txwq_websocket(cid.strip())
                    
                    if sgf_text:
                        st.success(status_msg)
                        fname = f"Live_Game_{datetime.datetime.now().strftime('%H%M')}.sgf"
                        st.download_button("💾 下载 SGF", sgf_text, file_name=fname)
                    else:
                        st.error(status_msg)
            else:
                st.warning("请先输入链接。")
                
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

        # --- 2.1 计算“上一局后的等级分变化 Delta” ---
        h_sorted = history_df.sort_values(['Name', 'Date']).copy()
        h_sorted['Prev_Rating'] = h_sorted.groupby('Name')['Rating'].shift(1)
        last_rows = h_sorted.groupby('Name').tail(1)[['Name', 'Rating', 'Prev_Rating']]
        last_rows['Delta'] = last_rows['Rating'] - last_rows['Prev_Rating']
        delta_df = last_rows[['Name', 'Delta']]

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
                       .merge(delta_df, on='Name', how='left'))
            full_df['Total_Games'] = full_df['Total_Games'].fillna(0).astype(int)
            full_df['Win_Rate'] = full_df['Win_Rate'].fillna('0.0%')
            full_df['Win_Count'] = full_df['Win_Count'].fillna(0).astype(int)
            full_df['Delta'] = full_df['Delta'].fillna(0)

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

                # 生成“变化”列（↑ 12 / ↓ 8 / —）
                def format_delta_cell(v):
                    try:
                        v = float(v)
                    except Exception:
                        return '—'
                    if v == 0:
                        return '—'
                    arrow = '↑' if v > 0 else '↓'
                    return f"{arrow} {abs(int(v))}"

                display_df['Delta'] = display_df['Delta'].apply(format_delta_cell)

                # 整理列名
                display_df = display_df[['Name', 'Rating', 'Delta', 'Total_Games', 'Win_Rate']]
                display_df.columns = ['选手', '等级分', '变化', '总局数', '总胜率']

                # 着色：涨分绿、跌分红
                def highlight_delta(val):
                    if isinstance(val, str):
                        if val.startswith('↑'):
                            return 'color: #16a34a;'  # 绿色
                        if val.startswith('↓'):
                            return 'color: #dc2626;'  # 红色
                    return ''

                styled = display_df.style.map(highlight_delta, subset=['变化'])

                st.dataframe(styled, width="stretch")
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
            st.dataframe(display_h2h[cols_to_show], width="stretch", height=400)

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
    edited = st.data_editor(
        recent_display,
        num_rows="fixed",
        hide_index=True,
        key="data_maintain_editor",
    )

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
