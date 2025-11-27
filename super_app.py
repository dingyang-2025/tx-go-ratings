import streamlit as st
import pandas as pd
import os
import altair as alt

# ==========================================
# 🔑 配置区
# ==========================================
FILE_PATH = "data.csv"

# ==========================================
# 🛠️ 辅助功能
# ==========================================

# 纯净版：只去空格，不再做ID映射
def standardize_name(input_name):
    if not input_name: return ""
    return str(input_name).strip()

# --- 1. ELO 算法 ---
def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_winner, rating_loser, k=32):
    expected_winner = calculate_expected_score(rating_winner, rating_loser)
    new_rating_winner = rating_winner + k * (1 - expected_winner)
    new_rating_loser = rating_loser + k * (0 - expected_winner)
    return new_rating_winner, new_rating_loser

# --- 2. 数据处理 ---
def load_data():
    columns = ["Date", "Player1", "Player2", "Winner", "Note1", "Note2"]
    
    if not os.path.exists(FILE_PATH):
        df = pd.DataFrame(columns=columns)
        df.to_csv(FILE_PATH, index=False)
    
    try:
        df = pd.read_csv(FILE_PATH)
        if df.empty: return pd.DataFrame(columns=columns)
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 兼容性处理：显示用的 Note 列
        if 'Note' not in df.columns:
            df['Note1'] = df['Note1'].fillna('')
            df['Note2'] = df['Note2'].fillna('')
            df['Note'] = df['Note1'] + ' | ' + df['Note2']
            
        return df
    except:
        return pd.DataFrame(columns=columns)

def save_game(date, p1, p2, winner, note1, note2):
    p1_std = standardize_name(p1)
    p2_std = standardize_name(p2)
    winner_std = standardize_name(winner)
    
    new_row = pd.DataFrame({
        "Date": [date], 
        "Player1": [p1_std], 
        "Player2": [p2_std], 
        "Winner": [winner_std], 
        "Note1": [note1],
        "Note2": [note2]
    })
    
    header = not os.path.exists(FILE_PATH)
    new_row.to_csv(FILE_PATH, mode='a', header=header, index=False)

def calculate_ratings(df, initial_rating=1500, k_factor=32):
    # 1. 准备容器
    ratings = {}  # 实时积分字典
    history = []  # 历史记录列表
    last_active = {} # 最后活跃时间

    # 2. 遍历每一行比赛数据
    for index, row in df.iterrows():
        # 获取人名并去除首尾空格（防止 "张三 " != "张三" 的情况）
        p1 = str(row['Player1']).strip()
        p2 = str(row['Player2']).strip()
        winner = str(row['Winner']).strip()
        date = row['Date']

        # --- 【修复核心】：自动初始化新选手 ---
        # 如果选手字典里还没有这个人，直接给初始分 1500
        if p1 not in ratings:
            ratings[p1] = initial_rating
        if p2 not in ratings:
            ratings[p2] = initial_rating
        
        # 更新活跃时间
        last_active[p1] = date
        last_active[p2] = date

        # --- 数据完整性检查 ---
        # 如果 Winner 是空的，或者是平局，或者Winner不在参赛者中
        if winner not in [p1, p2]:
            # print(f"警告：第 {index} 行数据异常，胜者 {winner} 不在选手 [{p1}, {p2}] 中，已跳过。")
            continue 

        # 确定败者
        loser = p2 if winner == p1 else p1

        # 获取当前分数 (此时因为上面已经做了初始化，绝对不会报错 KeyError 了)
        r_w = ratings[winner]
        r_l = ratings[loser]

        # 计算期望胜率
        e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        e_l = 1 / (1 + 10 ** ((r_w - r_l) / 400))

        # 更新分数
        new_r_w = r_w + k_factor * (1 - e_w)
        new_r_l = r_l + k_factor * (0 - e_l)

        ratings[winner] = new_r_w
        ratings[loser] = new_r_l

        # 记录历史
        history.append({
            'Date': date,
            'Player': winner,
            'Rating': new_r_w,
            'Opponent': loser,
            'Result': 'Win',
            'Note1': row.get('Note1', ''),
            'Note2': row.get('Note2', '')
        })
        history.append({
            'Date': date,
            'Player': loser,
            'Rating': new_r_l,
            'Opponent': winner,
            'Result': 'Loss',
            'Note1': row.get('Note1', ''),
            'Note2': row.get('Note2', '')
        })

    # 转换为 DataFrame
    history_df = pd.DataFrame(history)
    return ratings, last_active, history_df

# --- 3. 统计分析模块 ---
def get_rival_analysis(player_name, df):
    my_games = df[(df['Player1'] == player_name) | (df['Player2'] == player_name)]
    stats = {} # {opp: [wins, total]}
    
    for _, row in my_games.iterrows():
        opp = row['Player2'] if row['Player1'] == player_name else row['Player1']
        is_win = 1 if row['Winner'] == player_name else 0
        
        if opp not in stats: stats[opp] = [0, 0]
        stats[opp][0] += is_win # wins
        stats[opp][1] += 1      # total

    results = []
    for opp, (w, t) in stats.items():
        results.append({
            'name': opp,
            'total': t,
            'wins': w,
            'win_rate': (w/t)*100
        })
    return results

# --- 4. 界面主逻辑 ---
st.set_page_config(page_title="公司围棋大脑", layout="wide")
st.title("Go Ratings & Stats 📊")

df = load_data()
ratings, last_active, history_df = calculate_ratings(df)

# 动态获取所有出现过的选手名单
all_known_players = set(df['Player1'].dropna().unique()) | set(df['Player2'].dropna().unique())
known_names = sorted(list(all_known_players))

# === 侧边栏：录入新对局 ===
with st.sidebar:
    st.header("📝 录入新对局")
    with st.form("add_game"):
        new_date = st.date_input("日期")
        
        p1 = st.selectbox("选手1 (Player1)", ["(请选择)"] + known_names + ["(手动输入)"], index=0)
        p2 = st.selectbox("选手2 (Player2)", ["(请选择)"] + known_names + ["(手动输入)"], index=0)
        
        if p1 == "(手动输入)": p1 = st.text_input("请输入选手1名字")
        if p2 == "(手动输入)": p2 = st.text_input("请输入选手2名字")
        
        winner_c = st.radio("胜者", ["选手1胜", "选手2胜"])
        
        note1 = st.text_input("赛事名称 (Note1)", placeholder="例如：12届腾赛")
        note2 = st.text_input("轮次 (Note2)", placeholder="例如：第一轮")
        
        if st.form_submit_button("提交"):
            if p1 and p2 and p1 != "(请选择)" and p2 != "(请选择)" and p1 != p2:
                final_win = p1 if winner_c == "选手1胜" else p2
                save_game(new_date, p1, p2, final_win, note1, note2)
                st.success(f"已保存：{p1} vs {p2}")
                st.rerun()
            else:
                st.error("请完整填写选手信息")

# === 排行榜 & 走势 ===
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("🏆 实时排行")
    active_check = st.checkbox("只看活跃 (近2年)", value=True)
    if ratings:
        rank_data = []
        now = pd.Timestamp.now()
        for n, s in ratings.items():
            # 活跃检测
            if active_check and (now - last_active.get(n, pd.Timestamp.min)).days > 730: 
                continue 
            rank_data.append({"选手": n, "分数": int(s)})
        if rank_data:
            rdf = pd.DataFrame(rank_data).sort_values("分数", ascending=False).reset_index(drop=True)
            rdf.index += 1
            st.dataframe(rdf, height=400, use_container_width=True)
        else:
            st.info("😴 暂无活跃选手")
    else:
        st.info("暂无数据")

with c2:
    st.subheader("📈 历史走势")
    if not history_df.empty:
        # 默认选择前5名
        top_players = [p for p, s in sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:5]]
        opts = st.multiselect("对比选手", ratings.keys(), default=top_players)
        if opts:
            cd = history_df[history_df['Name'].isin(opts)]
            ymin, ymax = cd['Rating'].min()-50, cd['Rating'].max()+50
            c = alt.Chart(cd).mark_line(point=True).encode(
                x='Date', 
                y=alt.Y('Rating', scale=alt.Scale(domain=[ymin, ymax])), 
                color='Name', 
                tooltip=['Date','Name','Rating']
            ).interactive()
            st.altair_chart(c, use_container_width=True)

st.divider()

# === 选手详细档案 ===
st.subheader("🔍 选手详细档案")
col_search, col_stats = st.columns([1, 3])

# 初始化 session_state 防止报错
if 'current_selected_player' not in st.session_state:
    st.session_state.current_selected_player = "(请选择)"

with col_search:
    target = st.selectbox(
        "选择选手查看详情:", 
        ["(请选择)"] + sorted(list(ratings.keys())), 
        key="current_selected_player"
    )

if target != "(请选择)":
    # 1. 基础数据
    my_games = df[(df['Player1'] == target) | (df['Player2'] == target)].sort_values("Date", ascending=False)
    total_games = len(my_games)
    wins = len(my_games[my_games['Winner'] == target])
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    curr_score = int(ratings.get(target, 1500))
    
    # 2. 历史极值
    my_history = history_df[history_df['Name'] == target].sort_values('Date')
    if not my_history.empty:
        peak_row = my_history.loc[my_history['Rating'].idxmax()]
        peak_score = int(peak_row['Rating'])
        peak_date = peak_row['Date'].strftime('%Y-%m-%d')
        
        low_row = my_history.loc[my_history['Rating'].idxmin()]
        low_score = int(low_row['Rating'])
        low_date = low_row['Date'].strftime('%Y-%m-%d')
    else:
        peak_score = low_score = curr_score
        peak_date = low_date = "N/A"

    # 3. 对手分析
    rival_data = get_rival_analysis(target, df)
    old_rivals = sorted(rival_data, key=lambda x: x['total'], reverse=True)[:3]
    nemesis = sorted([r for r in rival_data if r['total'] >= 2], key=lambda x: x['win_rate'])[:3]
    preys = sorted([r for r in rival_data if r['total'] >= 2], key=lambda x: x['win_rate'], reverse=True)[:3]

    with col_stats:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("当前等级分", curr_score)
        m2.metric("巅峰等级分", peak_score, delta=f"{peak_date}")
        m3.metric("最低等级分", low_score, delta=f"{low_date}", delta_color="inverse")
        m4.metric("总对局数", f"{total_games} 局")
        m5.metric("总胜率", f"{win_rate:.1f}%")
        
        st.divider()
        
        c_rival, c_nemesis, c_prey = st.columns(3)
        
        def format_list(data_list):
            if not data_list: return "无数据"
            txt = ""
            for i, r in enumerate(data_list):
                txt += f"**{i+1}. {r['name']}** ({r['total']}局, 胜{r['win_rate']:.0f}%)\n\n"
            return txt

        with c_rival:
            st.markdown("#### 🤝 老对手 (交手最多)")
            st.markdown(format_list(old_rivals))
            
        with c_nemesis:
            st.markdown("#### ☠️ 苦手 (胜率最低)")
            st.caption("*(对局数≥2)*")
            st.markdown(format_list(nemesis))
            
        with c_prey:
            st.markdown("#### 🍲 下手 (胜率最高)")
            st.caption("*(对局数≥2)*")
            st.markdown(format_list(preys))

    st.divider()
    
    # --- 个人对局记录 ---
    st.markdown(f"#### 📜 {target} 完整对局记录")
    if not my_games.empty:
        display_games = my_games.rename(columns={
            "Date": "日期", "Player1": "选手1", "Player2": "选手2", "Winner": "获胜者", "Note": "备注"
        })
        display_games['日期'] = display_games['日期'].dt.strftime('%Y-%m-%d')
        cols_to_show = ["日期", "选手1", "选手2", "获胜者", "备注"]
        st.dataframe(display_games[cols_to_show], use_container_width=True)
    else:
        st.info("暂无对局记录")

st.divider()
st.subheader("📜 全公司完整对局记录")
if not df.empty:
    full_display = df.sort_values("Date", ascending=False).rename(columns={
        "Date": "日期", "Player1": "选手1", "Player2": "选手2", "Winner": "获胜者", "Note": "备注"
    })
    full_display['日期'] = full_display['日期'].dt.strftime('%Y-%m-%d')
    cols_to_show = ["日期", "选手1", "选手2", "获胜者", "备注"]
    st.dataframe(full_display[cols_to_show], use_container_width=True, height=500)
