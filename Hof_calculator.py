import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from io import BytesIO

sns.set(style="whitegrid")

# === 1️⃣ 예제 훈련 데이터 ===
train_data = [
    ("Adrian Beltre",208.7,82.7,1),
    ("Joe Mauer",92.0,54.4,1),
    ("Chipper Jones",160.0,70.4,1),
    ("Jim Thome",180.0,75.9,1),
    ("Scott Rolen",150.0,70.7,1),
    ("Larry Walker",145.0,71.0,1),
    ("Billy Wagner",95.0,62.0,0),
    ("Jeff Kent",122.0,75.0,0),
    ("Buster Posey",79.0,60.0,1),
    ("Yadier Molina",169.0,91.0,1),
    ("Adam Wainwright",86.0,72.0,0),
    ("Aaron Judge",110.0,75.0,1),
    ("Cole Hamels",57.0,60.0,0),
    ("Shin-Soo Choo",14.0,40.0,0),
    ("Ryu Hyun-jin",14.0,28.0,0)
]
df_train = pd.DataFrame(train_data, columns=["name","HOFm","WAR","elected"])
X, y = df_train[["HOFm","WAR"]].values, df_train["elected"].values
model = LogisticRegression(max_iter=1000).fit(X, y)

# === 2️⃣ 데이터 수집 함수 ===
def search_player(name):
    try:
        query = name.replace(" ","+")
        url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text,"html.parser")
        items = soup.select("div.search-item")
        players=[]
        for item in items:
            tag = item.find("a")
            if tag:
                href = tag["href"]
                desc = item.text.replace(tag.text,"").strip()
                players.append((tag.text, desc, f"https://www.baseball-reference.com{href}"))
        return players
    except:
        return []

def get_stats(url):
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text,"html.parser")
        def find(label):
            tag = soup.find("strong",string=label)
            if not tag: return None
            span = tag.find_next("span")
            try: return float(span.text)
            except: return None
        hofm = find("Hall of Fame Monitor")
        war = find("WAR")
        war7 = war*0.9 if war else None
        jaws = (hofm+war7)/2 if hofm and war7 else None
        # 시즌별 예제 데이터
        seasons = list(range(1,11))
        war_season = [war*0.1*i for i in seasons] if war else [0]*10
        ops_season = [50+5*i for i in seasons]
        era_season = [3.0+0.1*i for i in seasons]
        return hofm, war, war7, jaws, seasons, war_season, ops_season, era_season
    except:
        return None,None,None,None,[],[],[],[]

def compute_similar(hofm, war):
    df_train["distance"] = euclidean_distances([[hofm, war]], df_train[["HOFm","WAR"]])[0]
    return df_train.sort_values("distance").head(5)

# === 3️⃣ Streamlit UI ===
st.set_page_config(page_title="HOF Real MLB Engine", layout="wide")
st.title("🏆 Hall of Fame Real MLB Engine")
st.markdown("선수 이름 입력 → HOF 확률 + JAWS/WAR7 + 유사 선수 + 시즌별 그래프 + PDF 보고서")

name = st.text_input("선수 이름:", "")

if name:
    players = search_player(name)
    if not players:
        st.error("검색 결과 없음")
    else:
        idx = 0
        if len(players) > 1:
            st.info("동명이인 선수 선택")
            options = [f"{p[0]} — {p[1]}" for p in players]
            selected = st.selectbox("선수 선택:", options)
            idx = options.index(selected)
        pname, desc, url = players[idx]

        hofm, war, war7, jaws, seasons, war_season, ops_season, era_season = get_stats(url)

        if hofm is None or war is None:
            st.error("데이터 없음")
        else:
            prob = model.predict_proba([[hofm, war]])[0,1]
            similar = compute_similar(hofm, war)

            tab1, tab2, tab3, tab4 = st.tabs(["요약","시즌 그래프","유사 선수","PDF 보고서"])

            with tab1:
                st.subheader(f"🧾 {pname} 예측 결과")
                st.write(f"HOF Monitor: {hofm}")
                st.write(f"WAR: {war}")
                st.write(f"WAR7: {war7:.1f}")
                st.write(f"JAWS: {jaws:.1f}")
                st.write(f"BBWAA 헌액 확률: {prob*100:.1f}%")
                st.markdown(f"[📎 Baseball Reference 링크]({url})")

            with tab2:
                st.subheader("📈 시즌별 WAR/OPS+/ERA+ 트렌드 (MLB 스타일)")
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(seasons, war_season, label="WAR", color="#1f77b4", marker="o", linewidth=2)
                ax.plot(seasons, ops_season, label="OPS+", color="#ff7f0e", marker="s", linewidth=2)
                ax.plot(seasons, era_season, label="ERA+", color="#2ca02c", marker="^", linewidth=2)
                ax.set_xlabel("시즌", fontsize=12)
                ax.set_ylabel("값", fontsize=12)
                ax.set_title(f"{pname} 시즌별 통계 트렌드", fontsize=14, fontweight="bold")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            with tab3:
                st.subheader("🔍 유사 선수 TOP 5")
                st.dataframe(similar[["name","HOFm","WAR","elected"]].rename(columns={
                    "name":"선수","HOFm":"HOFm","WAR":"WAR","elected":"명전(1=헌액)"
                }))

            with tab4:
                st.subheader("📄 PDF 보고서 생성")
                if st.button("PDF 생성 및 다운로드"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial","B",18)
                    pdf.set_text_color(30,30,60)
                    pdf.cell(0,10,f"HOF Real MLB Report: {pname}", ln=True, align="C")
                    pdf.set_font("Arial","",12)
                    pdf.ln(5)
                    pdf.cell(0,8,f"HOF Monitor: {hofm}", ln=True)
                    pdf.cell(0,8,f"WAR: {war}", ln=True)
                    pdf.cell(0,8,f"WAR7: {war7:.1f}", ln=True)
                    pdf.cell(0,8,f"JAWS: {jaws:.1f}", ln=True)
                    pdf.cell(0,8,f"BBWAA 헌액 확률: {prob*100:.1f}%", ln=True)
                    pdf.ln(5)
                    pdf.cell(0,8,"유사 선수 TOP 5", ln=True)
                    for _, row in similar.iterrows():
                        pdf.cell(0,6,f"{row['name']} | HOFm: {row['HOFm']} | WAR: {row['WAR']} | 헌액:{row['elected']}", ln=True)
                    
                    # 그래프를 PNG로 저장 후 PDF에 깔끔하게 삽입
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    pdf.image(buf, x=15, y=120, w=180) # x, y 위치와 너비 조정
                    pdf_bytes = pdf.output(dest='S').encode('latin1')

                    st.download_button(
                        label="✅ PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"{pname.replace(' ','_')}_HOF_Report.pdf",
                        mime="application/pdf"
                    )