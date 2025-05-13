import streamlit as st  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64
import shap
import lime
import lime.lime_tabular

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes (1).csv")

data = load_data()
columns = data.columns.tolist()

# ---------------------- Preprocessing ----------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ---------------------- Background Styling ----------------------
def set_background(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

# ---------------------- PDF Generator ----------------------
def generate_pdf(user_inputs, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Patient Input Parameters:", ln=True)
    pdf.ln(5)

    for key, value in user_inputs.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Prediction Result: {'Diabetes (1)' if prediction else 'No Diabetes (0)'}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, """\
Medical Analysis:
Several factors influence the prediction of diabetes:
- High glucose levels (Glucose) are a strong indicator of diabetes.
- Higher BMI indicates obesity which increases diabetes risk.
- Insulin levels, skin thickness, and age are also significant.
- A higher number of pregnancies can increase the risk in women.

This report is system-generated and based on logistic regression analysis of historical data.
Consult a physician for clinical interpretation.
    """)

    # Output PDF to bytes
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

def get_pdf_download_link(pdf_bytes, filename="diabetes_report.pdf"):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download PDF Report</a>'
    return href

# ---------------------- SHAP Explanation ----------------------
def explain_shap(model, X_scaled):
    # The explainer expects a model and the data it's trained on (X_scaled)
    explainer = shap.Explainer(model, X_scaled)  # Ensure correct model format
    shap_values = explainer(X_scaled)  # Generate SHAP values for the dataset

    # Creating a figure for SHAP plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size for better visualization
    shap.summary_plot(shap_values, X_scaled, show=False)  # Set show=False to avoid auto-display
    st.pyplot(fig)  # Explicitly pass the figure object here
    return shap_values

# ---------------------- LIME Explanation ----------------------
def explain_lime(model, X_train, X_test, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['No Diabetes', 'Diabetes'],
        mode='classification'
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba)
    
    # LIME Explanation List
    lime_explanation = exp.as_list()  # This will give a list of feature contributions
    
    # Plot LIME Explanation
    fig, ax = plt.subplots(figsize=(8, 6))
    features = [item[0] for item in lime_explanation]
    contributions = [item[1] for item in lime_explanation]
    
    ax.barh(features, contributions, color='lightblue')
    ax.set_xlabel('Contribution to Prediction')
    ax.set_title('LIME Feature Contributions')

    # Display the plot
    st.pyplot(fig)
    
    return lime_explanation

# ---------------------- Login Page ----------------------
def login_page():
    set_background("https://www.ch-chateaudun.fr/wp-content/uploads/2023/01/laboratoire-gants-echantillons-sang-freepik-scaled.jpg")
    st.title("HeAlth +  Labs")
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ---------------------- Main App ----------------------
def main_app():
    set_background("https://i.pinimg.com/originals/ee/d8/19/eed8191a10a1a2beb74a3526a2d2344d.jpg")
    st.title("🩺 Diabetes Dashboard & Predictor")

    # Logout
    with st.sidebar:
        st.markdown("## 🔒 Session")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    menu = st.sidebar.radio("Navigation", ["Home", "Data Summary", "Visualizations", "Predict Outcome"])

    # ---------------------- Home ----------------------
    if menu == "Home":
        st.markdown("""\
            ## 👋 Welcome!
            This interactive dashboard allows you to:
            - 🔍 Explore diabetes data
            - 📊 Visualize health patterns
            - 🤖 Predict diabetes outcome using a trained Logistic Regression model
        """)
        if st.checkbox("Show Raw Data"):
            st.dataframe(data)

    # ---------------------- Summary ----------------------
    elif menu == "Data Summary":
        st.subheader("📊 Dataset Summary Statistics")
        st.dataframe(data.describe())

        st.subheader("📌 Class Distribution")
        class_counts = data.iloc[:, -1].value_counts().rename({0: 'No Diabetes', 1: 'Diabetes'})
        st.bar_chart(class_counts)

    # ---------------------- Visualizations ----------------------
    elif menu == "Visualizations":
        st.subheader("📈 Scatterplot")
        x_axis = st.selectbox("Select X-axis", columns, key="xaxis")
        y_axis = st.selectbox("Select Y-axis", columns, key="yaxis")

        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=data.columns[-1], ax=ax)
        st.pyplot(fig)

        st.subheader("📉 Histogram")
        selected_col = st.selectbox("Select Column for Histogram", columns, key="hist")
        fig, ax = plt.subplots()
        sns.histplot(data[selected_col], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------------- Prediction + PDF + SHAP + LIME ----------------------
    elif menu == "Predict Outcome":
        st.subheader("🧮 Predict Diabetes Based on Your Inputs")
        user_inputs = {}
        for col in columns[:-1]:  # Exclude target column
            user_inputs[col] = st.number_input(f"{col}", value=float(data[col].mean()))

        if st.button("Predict Outcome"):
            user_data = scaler.transform([list(user_inputs.values())])
            prediction = model.predict(user_data)[0]
            result = f"🩺 Predicted Outcome: {'Diabetes (1)' if prediction else 'No Diabetes (0)'}"
            st.success(result)

            # SHAP Explanation
            shap_values = explain_shap(model, X_scaled)  # SHAP explanation

            # LIME Explanation + Plot
            lime_explanation = explain_lime(model, X_scaled, user_data, columns[:-1])  # LIME explanation with plot
            st.write("### LIME Explanation:")
            for feature, contribution in lime_explanation:
                st.write(f"Feature: {feature}, Contribution: {contribution:.4f}")  # Display LIME explanation

            # PDF report generation
            pdf_bytes = generate_pdf(user_inputs, prediction)
            st.markdown(get_pdf_download_link(pdf_bytes), unsafe_allow_html=True)

# ---------------------- Controller ----------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
else:
    main_app()
