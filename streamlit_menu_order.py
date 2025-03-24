import sounddevice as sd
import numpy as np
import noisereduce as nr
import streamlit as st
import wave
import whisper
import re
import io

# Define your menu items
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


import streamlit as st
import pandas as pd

# Convert menu to DataFrame
df = pd.DataFrame(menu)

st.title("🍽️ Restaurant Menu")

# Display styled menu (Excel-like)
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

# Audio Recording parameters
SAMPLE_RATE = 16000
audio_file = "recorded_order.wav"

# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# model = load_menu = st.cache_resource(load_menu)

def record_audio(output_file, sample_rate=16000):
    duration = 30  # Max duration (seconds)
    st.write("🎤 Recording... Click STOP to end earlier.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    recording = recording.flatten()
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    st.success(f"✅ Recording saved to {filename}")


# Recording logic with start and stop button
if 'recording' not in st.session_state:
    st.session_state.recording = False

if st.button('🎤 Start Recording'):
    st.session_state.recording = True

if st.session_state.recording:
    if st.button('⏹️ Stop Recording'):
        sd.stop()
        st.session_state.recording = False
        st.success("✅ Recording stopped.")

if st.session_state.recording:
    record_audio(audio_file)
    # record_audio(SAMPLE_RATE, audio_file)

######################################
import openai
import pandas as pd
import io


# Set your OpenAI API Key here securely
# openai.api_key = ""

openai.api_key = st.secrets["OPENAI_API_KEY"]

def correct_transcription(transcribed_text, df):
    # Convert DataFrame to dictionary for easy lookup
    menu_dict = {row["Dish"].lower(): row["Price (INR)"] for _, row in df.iterrows()}

    # Create menu reference for LLM
    menu_reference = "\n".join([f"{dish}, ₹{price}" for dish, price in menu_dict.items()])

    prompt = f"""
    You are an AI that processes restaurant orders.

    **Step 1:** Correct the transcribed order based on the available menu items below:
    ```
    {menu_reference}
    ```

    **Step 2:** After correcting, format the final order strictly as a CSV table with the following columns:
    - **Quantity**
    - **Dish Name** (matched from menu)
    - **Price per item**
    - **Total Price per dish** (Quantity * Price per item)

    **Step 3:** Compute the final bill total.

    **Example Output (Strictly CSV Format):**
    ```
    Quantity,Dish Name,Price per item,Total Price per dish
    2,Paneer Butter Masala,200,400
    1,Garlic Naan,40,40
    3,Gulab Jamun,60,180
    ,Total Amount,,620
    ```

    **Transcribed Order to Correct:**
    ```
    {transcribed_text}
    ```

    Return the output strictly as a CSV table, ensuring all dishes match the menu.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant processing restaurant orders."},
                  {"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response["choices"][0]["message"]["content"].strip()


import streamlit as st
import pandas as pd
import io
import re
import random  # For estimating preparation time

# ✅ Initialize session state variables
if "order_confirmed" not in st.session_state:
    st.session_state.order_confirmed = False
if "order_df" not in st.session_state:
    st.session_state.order_df = None
if "total_amount" not in st.session_state:
    st.session_state.total_amount = 0
if "re_record" not in st.session_state:
    st.session_state.re_record = False

# ✅ If user pressed "No, Re-record Order", reset everything
if st.session_state.re_record:
    st.session_state.order_confirmed = False
    st.session_state.order_df = None
    st.session_state.total_amount = 0
    st.session_state.re_record = False  # Reset flag

    # ✅ Show only "Record Order" button
    if st.button("🎤 Re-record Order"):
        st.session_state.re_record = False  # Reset again to prevent loop
        st.rerun()
    st.stop()  # 🚀 Stop execution to ensure only "Record Order" is displayed

if st.button("✅ Show Final Order"):
    with st.spinner("Processing order..."):
        # ✅ Get actual LLM output
        model = load_whisper_model()
        transcription = model.transcribe(audio_file)['text']
        corrected_csv = correct_transcription(transcription, df)  # 👈 Fetch actual LLM output

        try:
            # ✅ Step 1: Preprocess CSV
            cleaned_csv = corrected_csv.strip()  
            cleaned_csv = re.sub(r'^```|```$', '', cleaned_csv)  
            cleaned_csv = re.sub(r'<NA>', '', cleaned_csv)  
            cleaned_csv = re.sub(r',+', ',', cleaned_csv)  
            cleaned_csv = re.sub(r'\n+', '\n', cleaned_csv)  

            # ✅ Step 2: Read CSV safely
            order_df = pd.read_csv(io.StringIO(cleaned_csv), dtype=str)

            # ✅ Step 3: Normalize column names (remove spaces & convert to lowercase)
            order_df.columns = order_df.columns.str.strip().str.lower()

            # ✅ Step 4: Remove fully empty rows
            order_df.dropna(how='all', inplace=True)

            # ✅ Step 5: Fix "Quantity" column (Remove unintended index numbers)
            if order_df.iloc[:, 0].str.isnumeric().all():
                order_df.iloc[:, 0] = pd.to_numeric(order_df.iloc[:, 0], errors="coerce")

            # ✅ Step 6: Ensure only ONE "Total Amount" row exists
            price_column = [col for col in order_df.columns if "total price" in col.lower()]
            if price_column:
                order_df[price_column[0]] = pd.to_numeric(order_df[price_column[0]], errors="coerce")
                total_amount = order_df[price_column[0]].sum()

                # ✅ Remove any existing incorrect "Total Amount" rows
                order_df = order_df[~order_df["dish name"].str.contains("Total Amount", na=False)]

                # ✅ Create the correct "Total Amount" row
                total_row = pd.DataFrame([["", "**Total Amount**", "", f"**₹{total_amount:.2f}**"]], columns=order_df.columns)

                # ✅ Append the correct row
                order_df = pd.concat([order_df, total_row], ignore_index=True)

            # ✅ Step 7: Store the order in session state
            st.session_state.order_df = order_df
            st.session_state.total_amount = total_amount
            st.session_state.order_confirmed = False  # Reset confirmation state

            # ✅ Step 8: Display the final order
            st.success("✅ Final Order")
            st.table(order_df)

        except Exception as e:
            st.error(f"⚠️ Error processing CSV: {e}")
            st.text_area("LLM Raw Output:", corrected_csv, height=200)

# ✅ Order Confirmation Step
if st.session_state.order_df is not None:
    st.write("🛒 **Would you like to confirm your order?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Place Order", disabled=st.session_state.order_confirmed):
            st.session_state.order_confirmed = True
            st.rerun()  # ✅ Refresh page to show the order placed message

    with col2:
        if st.button("🔁 No, Re-record Order", disabled=st.session_state.order_confirmed):
            st.session_state.re_record = True  # ✅ Set flag to restart process
            st.rerun()  # ✅ Restart the process and show only "Record Order"

# ✅ If Order is Confirmed, Show Estimated Time
if st.session_state.order_confirmed:
    st.success("🎉 Your Order is Placed! 🍽️")
    
    # ✅ Estimate preparation time (random between 10-30 minutes)
    prep_time = random.randint(10, 30)
    st.markdown(f"⏳ **Approximate Time to Get Your Order Ready: {prep_time} minutes 😊. Enjoy Music 🎧 **")
