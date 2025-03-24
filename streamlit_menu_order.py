import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import random
import wave
import openai
import whisper
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# -------------------- Menu --------------------
menu = [
    {"Category": "Appetizers", "Dish": "Vegetable Samosa", "Price (INR)": 50},
    {"Category": "Appetizers", "Dish": "Paneer Tikka", "Price (INR)": 150},
    {"Category": "Appetizers", "Dish": "Chicken Seekh Kebab", "Price (INR)": 180},
    {"Category": "Appetizers", "Dish": "आलू टिक्की (Aloo Tikki)", "Price (INR)": 70},
    {"Category": "Snacks", "Dish": "Masala Dosa", "Price (INR)": 100},
    {"Category": "Snacks", "Dish": "Paneer Masala Dosa", "Price (INR)": 120},
    {"Category": "Snacks", "Dish": "पाव भाजी (Pav Bhaji)", "Price (INR)": 110},
    {"Category": "Main Course - Vegetarian", "Dish": "Paneer Butter Masala", "Price (INR)": 200},
    {"Category": "Main Course - Vegetarian", "Dish": "Dal Makhani", "Price (INR)": 180},
    {"Category": "Main Course - Vegetarian", "Dish": "छोले भटूरे (Chole Bhature)", "Price (INR)": 130},
    {"Category": "Main Course - Non-Vegetarian", "Dish": "Butter Chicken", "Price (INR)": 250},
    {"Category": "Main Course - Non-Vegetarian", "Dish": "Lamb Rogan Josh", "Price (INR)": 280},
    {"Category": "Main Course - Non-Vegetarian", "Dish": "चिकन करी (Chicken Curry)", "Price (INR)": 240},
    {"Category": "Breads", "Dish": "Naan", "Price (INR)": 30},
    {"Category": "Breads", "Dish": "Garlic Naan", "Price (INR)": 40},
    {"Category": "Breads", "Dish": "Tandoori Roti", "Price (INR)": 20},
    {"Category": "Breads", "Dish": "आलू पराठा (Aloo Paratha)", "Price (INR)": 60},
    {"Category": "Desserts", "Dish": "Gulab Jamun", "Price (INR)": 60},
    {"Category": "Desserts", "Dish": "Rasgulla", "Price (INR)": 60},
    {"Category": "Desserts", "Dish": "रबड़ी (Rabri)", "Price (INR)": 80},
]

df = pd.DataFrame(menu)

st.title("🍽️ Restaurant Menu")

st.dataframe(
    df.style.set_properties(**{
        'text-align': 'left',
        'border': '1px solid black',
        'padding': '10px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#FFDDC1'), 
            ('color', '#000000'),
            ('font-size', '16px'),
            ('text-align', 'center')
        ]},
        {'selector': 'td', 'props': [('font-size', '14px')]}
    ]),
    use_container_width=True,
    hide_index=True
)

st.divider()

# -------------------- Whisper Model --------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# -------------------- WebRTC Audio Recorder --------------------
class AudioProcessor:
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)
# -------------------- Transcription + LLM Correction --------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

def correct_transcription(transcribed_text, df):
    menu_dict = {row["Dish"].lower(): row["Price (INR)"] for _, row in df.iterrows()}
    menu_reference = "\n".join([f"{dish}, ₹{price}" for dish, price in menu_dict.items()])

    prompt = f"""
    You are an AI that processes restaurant orders.

    **Step 1:** Correct the transcribed order based on the available menu items below:
    ```
    {menu_reference}
    ```

    **Step 2:** After correcting, format the final order strictly as a CSV table with the following columns:
    Quantity,Dish Name,Price per item,Total Price per dish

    **Step 3:** Add the final bill as the last row like:
    ,Total Amount,,XXX

    **Transcribed Order to Correct:**
    {transcribed_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant processing restaurant orders."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response["choices"][0]["message"]["content"].strip()

# -------------------- Session State --------------------
if "order_df" not in st.session_state:
    st.session_state.order_df = None
if "order_confirmed" not in st.session_state:
    st.session_state.order_confirmed = False

# -------------------- Transcribe and Process --------------------
if ctx.audio_processor and st.button("📝 Transcribe & Process Order"):
    if ctx.audio_processor.recorded_frames:
        # Save audio
        audio_data = np.concatenate(ctx.audio_processor.recorded_frames, axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with st.spinner("🔍 Transcribing..."):
                result = model.transcribe(f.name)
                transcription = result["text"]
                st.markdown("**📝 Transcribed Text:**")
                st.code(transcription)

                corrected_csv = correct_transcription(transcription, df)

                try:
                    # Clean and parse CSV
                    cleaned_csv = re.sub(r'^```|```$', '', corrected_csv.strip())
                    cleaned_csv = re.sub(r'<NA>', '', cleaned_csv)
                    cleaned_csv = re.sub(r',+', ',', cleaned_csv)
                    cleaned_csv = re.sub(r'\n+', '\n', cleaned_csv)

                    order_df = pd.read_csv(io.StringIO(cleaned_csv), dtype=str)
                    order_df.columns = order_df.columns.str.strip().str.lower()
                    order_df.dropna(how='all', inplace=True)
                    price_column = [col for col in order_df.columns if "total price" in col.lower()]
                    if price_column:
                        order_df[price_column[0]] = pd.to_numeric(order_df[price_column[0]], errors="coerce")
                        total_amount = order_df[price_column[0]].sum()
                        order_df = order_df[~order_df["dish name"].str.contains("Total Amount", na=False)]
                        total_row = pd.DataFrame([["", "Total Amount", "", f"₹{total_amount:.2f}"]], columns=order_df.columns)
                        order_df = pd.concat([order_df, total_row], ignore_index=True)

                    st.session_state.order_df = order_df
                    st.session_state.order_confirmed = False
                    st.success("✅ Final Order")
                    st.table(order_df)

                except Exception as e:
                    st.error(f"⚠️ Error processing CSV: {e}")
                    st.text_area("LLM Output:", corrected_csv, height=200)

# -------------------- Confirm Order --------------------
if st.session_state.order_df is not None and not st.session_state.order_confirmed:
    st.write("🛒 **Would you like to confirm your order?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Place Order"):
            st.session_state.order_confirmed = True
            st.rerun()
    with col2:
        if st.button("🔁 No, Re-record Order"):
            st.session_state.order_df = None
            st.session_state.order_confirmed = False
            ctx.audio_processor.recorded_frames.clear()
            st.rerun()

# -------------------- Order Confirmed Message --------------------
if st.session_state.order_confirmed:
    st.success("🎉 Your Order is Placed! 🍽️")
    prep_time = random.randint(10, 30)
    st.markdown(f"⏳ **Your order will be ready in ~{prep_time} minutes. Enjoy Music 🎧**")
