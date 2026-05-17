import streamlit as st
import sqlite3
import pandas as pd
from PIL import Image
import os


DB_PATH = "database/events.db"


st.set_page_config(
    page_title="ExamEye Dashboard",
    layout="wide"
)

st.title("🎓 ExamEye Monitoring Dashboard")


def load_events():
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT *
        FROM events
        ORDER BY id DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


events_df = load_events()


st.subheader("📋 Suspicious Events")

st.dataframe(events_df)


st.subheader("📸 Event Screenshots")

for _, row in events_df.iterrows():

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:

        screenshot_path = row["screenshot_path"]

        if os.path.exists(screenshot_path):

            image = Image.open(screenshot_path)

            st.image(image, width=300)

    with col2:

        st.write(f"### Student ID: {row['student_id']}")
        st.write(f"**Event:** {row['event_type']}")
        st.write(f"**Timestamp:** {row['timestamp']}")
        st.write(f"**Confidence:** {row['confidence']:.2f}")