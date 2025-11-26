import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import altair as alt

# ==========================================
# 🔑 配置区
# ==========================================
API_KEY = "这里填入你的OpenAI_Key"
FILE_PATH = "data.csv"

# ==========================================
# 📋 标准选手字典 (自动纠错核心)
# ==========================================
PLAYER_MAP = {
    '苏洋': 'youngy1997',
    '丁阳': '伶汀洋',
    '严修华': '棋若有情',
    '于川': '睡着的鱼大',
    '俞安彤': '南窗寄傲00',
    '刁寿钧': 'cachediao',
    '刘博东': 'jingleliu',
    '刘天一': 'liutian111',
    '卫然': 'randomness',
    '叶子鹏': 'Tay1203',
    '吕骥图': '修多阁下',
    '周子祺': 'fredlls',
    '周杨杰': '翻墙',
    '周淼': 'miozhou',
    '姚力涛': '幸福De小涛',
    '孙晨阳': '酸菜鱼同学',
    '张刚毅': '不下官子',
    '张家齐': 'seidemi',
    '彭天佐': '刀疤帮老五',
    '戴南': 'dainan2021',
    '曹易伦': '八索话人间',
    '朱磊': '围棋新生s',
    '朴乘志': '南山豆蔬菜',
    '李奇林': 'Modricc',
    '李春朔': '抵达14236',
    '李林': '256785',
    '李火荣': 'star022',
    '林琦慧': '三位不一体',
    '沈希阳': '烟花落寞',
    '洪时豪': '猴吃桃',
    '潘肇程': '南蔷北笙',
    '王周源': '源well',
    '王天奇': '大傻砸',
    '王早': '王早',
    '王昊': '骆驼祥19',
    '王福臣': 'sai5go',
    '王行健': '棋道中和',
    '申笑铭': 'AbyssLaugh',
    '秦亦周': 'lvver',
    '罗大为': 'dawidluo',
    '肖罗杰': '不灭的圣火',
    '肖越': 'Moon、Sai',
    '蒙锐': 'MMMrrr',
    '蔡江东': 'cjdbehum',
    '郑楷': 'dijjcnfjij',
    '金川杰': 'coolhead',
    '闫书染': 'SRAN皮皮',
    '陈愚夫': '織水信夫',
    '陈新星': '小一艺',
    '陈泽友': 'v211413371',
    '陈翔': 'joshuaxchen',
    '高一君': '不断地学',
    '黄博阳': 'ablehuang',
}

# --- 辅助功能：标准化名字 ---
def standardize_name(input_name):
    if not input_name: return ""
    name_str = str(input_name).strip()
    if name_str in PLAYER_MAP:
        return f"{name_str} ({PLAYER_MAP[name_str]})"
    for std_name, std_id in PLAYER_MAP.items():
        if std_id in name_str: 
            return f"{std_name} ({std_id})"
    return name_str

# --- 0. 状态初始化 ---
if 'current_selected_player' not in st.session_state:
    st.session_state.current_selected_player = "(请选择)"

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
    if not os.path.exists(FILE_PATH):
        df = pd.DataFrame(columns=["Date", "Black", "White", "Winner", "Note"])
        df.to_csv(FILE_PATH, index=False)
    try:
        df = pd.read_csv(FILE_PATH)
        if df.empty: return pd.DataFrame(columns=["Date", "Black", "White", "Winner", "Note"])
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return pd.DataFrame(columns=["Date", "Black", "White", "Winner", "Note"])

def save_game(date, p1, p2, winner, note):
    p1_std = standardize_name(p1)
    p2_std = standardize_name(p2)
    winner_std = standardize_name(winner)
    new_row = pd.DataFrame({"Date": [date], "Black": [p1_std], "White": [p2_std], "Winner": [winner_std], "Note": [note]})
    header = not os.path.exists(FILE_PATH)
    new_row.to_csv(FILE_PATH, mode='a', header=header, index=False)

def calculate_ratings(df):
    current_ratings = {}
    last_active = {}
    history = []
    
    if df.empty: return {}, {}, pd.DataFrame()

    for _, row in df.sort_values('Date').iterrows():
        black, white, winner, date = row['Black'], row['White'], row['Winner'], row['Date']
        
        if black not in current_ratings: current_ratings[black] = 1500
        if white not in current_ratings: current_ratings[white] = 1500
        
        loser = white if winner == black else black
        last_active[black] = date
        last_active[white] = date

        r_w, r_l = current_ratings[winner], current_ratings[loser]
        new_r_w, new_r_l = update_elo(r_w, r_l, k=32)
        current_ratings[winner], current_ratings[loser] = new_r_w, new_r_l
        
        history.append({'Date': date, 'Name': winner, 'Rating': new_r_w})
        history.append({'Date': date, 'Name': loser, 'Rating': new_r_l})
        
    return current_ratings, last_active, pd.DataFrame(history)

# --- 3. 统计分析模块 ---
def get_comprehensive_stats(ratings, df):
    # 简版统计，用于 AI 对话上下文
    summary_lines = []
    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for player, score in sorted_players:
        wins_df = df[df['Winner'] == player]
        total_wins = len(wins_df)
        summary_lines.append(f"选手:{player}|分:{int(score)}|总胜:{total_wins}")
    return "\n".join(summary_lines)

def get_rival_analysis(player_name, df):
    """
    计算老对手、苦手、下手
    返回格式: [ {'name':对手, 'total':局数, 'win_rate':胜率, 'wins':胜局}, ... ]
    """
    my_games = df[(df['Black'] == player_name) | (df['White'] == player_name)]
    stats = {} # {opp: [wins, total]}
    
    for _, row in my_games.iterrows():
        opp = row['White'] if row['Black'] == player_name else row['Black']
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

# === 侧边栏：录入新对局 ===
with st.sidebar:
    st.header("📝 录入新对局")
    with st.form("add_game"):
        new_date = st.date_input("日期")
        known_names = sorted(list(PLAYER_MAP.keys())) 
        p1 = st.selectbox("黑方 (Black)", ["(请选择)"] + known_names + ["(手动输入)"], index=0)
        p2 = st.selectbox("白方 (White)", ["(请选择)"] + known_names + ["(手动输入)"], index=0)
        if p1 == "(手动输入)": p1 = st.text_input("请输入黑方名字")
        if p2 == "(手动输入)": p2 = st.text_input("请输入白方名字")
        winner_c = st.radio("胜者", ["黑方胜", "白方胜"])
        note = st.text_input("备注 (例如：12届腾赛 | 第一轮)")
        
        if st.form_submit_button("提交"):
            if p1 and p2 and p1 != "(请选择)" and p2 != "(请选择)" and p1 != p2:
                final_win = p1 if winner_c == "黑方胜" else p2
                save_game(new_date, p1, p2, final_win, note)
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
        opts = st.multiselect("对比", ratings.keys(), default=list(ratings.keys())[:5])
        if opts:
            cd = history_df[history_df['Name'].isin(opts)]
            ymin, ymax = cd['Rating'].min()-50, cd['Rating'].max()+50
            c = alt.Chart(cd).mark_line(point=True).encode(x='Date', y=alt.Y('Rating', scale=alt.Scale(domain=[ymin, ymax])), color='Name', tooltip=['Date','Name','Rating']).interactive()
            st.altair_chart(c, use_container_width=True)

st.divider()

# === 选手详细档案 (重构版) ===
st.subheader("🔍 选手详细档案")
col_search, col_stats = st.columns([1, 3])

with col_search:
    target = st.selectbox(
        "选择选手查看详情:", 
        ["(请选择)"] + sorted(list(ratings.keys())), 
        key="current_selected_player"
    )

if target != "(请选择)":
    # 1. 基础数据计算
    my_games = df[(df['Black'] == target) | (df['White'] == target)].sort_values("Date", ascending=False)
    total_games = len(my_games)
    wins = len(my_games[my_games['Winner'] == target])
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    curr_score = int(ratings.get(target, 1500))
    
    # 2. 历史极值计算
    my_history = history_df[history_df['Name'] == target].sort_values('Date')
    if not my_history.empty:
        # 巅峰
        peak_row = my_history.loc[my_history['Rating'].idxmax()]
        peak_score = int(peak_row['Rating'])
        peak_date = peak_row['Date'].strftime('%Y-%m-%d')
        # 最菜 (最低)
        low_row = my_history.loc[my_history['Rating'].idxmin()]
        low_score = int(low_row['Rating'])
        low_date = low_row['Date'].strftime('%Y-%m-%d')
    else:
        peak_score = low_score = curr_score
        peak_date = low_date = "N/A"

    # 3. 对手分析
    rival_data = get_rival_analysis(target, df)
    # A. 老对手 (局数最多)
    old_rivals = sorted(rival_data, key=lambda x: x['total'], reverse=True)[:3]
    # B. 苦手 (局数>=2, 胜率最低 -> 升序)
    nemesis = sorted([r for r in rival_data if r['total'] >= 2], key=lambda x: x['win_rate'])[:3]
    # C. 下手 (局数>=2, 胜率最高 -> 降序)
    preys = sorted([r for r in rival_data if r['total'] >= 2], key=lambda x: x['win_rate'], reverse=True)[:3]

    with col_stats:
        # --- 第一行：核心指标 ---
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("当前等级分", curr_score)
        m2.metric("巅峰等级分", peak_score, delta=f"{peak_date}")
        m3.metric("最低等级分", low_score, delta=f"{low_date}", delta_color="inverse")
        m4.metric("总对局数", f"{total_games} 局")
        m5.metric("总胜率", f"{win_rate:.1f}%")
        
        st.divider()
        
        # --- 第二行：三大榜单 ---
        c_rival, c_nemesis, c_prey = st.columns(3)
        
        def format_list(data_list):
            if not data_list: return "无数据"
            txt = ""
            for i, r in enumerate(data_list):
                # 格式: 1. 张三 (10局, 胜40%)
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
    
    # --- 底部：完整对局记录 ---
    st.markdown(f"#### 📜 {target} 完整对局记录")
    if not my_games.empty:
        display_games = my_games.rename(columns={
            "Date": "日期", "Black": "黑方", "White": "白方", "Winner": "获胜者", "Note": "备注"
        })
        # 格式化日期列，只显示 YYYY-MM-DD
        display_games['日期'] = display_games['日期'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_games, use_container_width=True)
    else:
        st.info("暂无对局记录")

st.divider()

# === AI 咨询 & 完整记录 ===
st.subheader("💬 AI 围棋咨询师")
user_q = st.text_input("您可以问我任何问题（例如：谁最有希望夺冠？谁是连胜王？俞安彤赢过谁？）")
if user_q:
    with st.spinner("AI 思考中..."):
        st.write(ask_ai_general(user_q, ratings, df))

st.divider()
st.subheader("📜 全公司完整对局记录")
if not df.empty:
    full_display = df.sort_values("Date", ascending=False).rename(columns={
        "Date": "日期", "Black": "黑方", "White": "白方", "Winner": "获胜者", "Note": "备注"
    })
    full_display['日期'] = full_display['日期'].dt.strftime('%Y-%m-%d')
    st.dataframe(full_display, use_container_width=True, height=500)
