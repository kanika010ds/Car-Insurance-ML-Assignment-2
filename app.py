
# ============================================================
# CLAIMLENS STUDIO - CAR INSURANCE CLAIM ANALYTICS
# ============================================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ClaimLens Studio | Insurance Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
MODELS = BASE_DIR / "models"


# ============================================================
# 2. REFERENCE MODEL PERFORMANCE
# ============================================================

# Previously measured scores on the original test dataset
REFERENCE = pd.DataFrame(
    [
        ["Logistic Regression", 0.8455, 0.9158, 0.7548, 0.7512, 0.7530, 0.6406],
        ["Decision Tree", 0.8395, 0.9001, 0.7812, 0.6778, 0.7259, 0.6163],
        ["KNN", 0.8145, 0.8871, 0.7000, 0.7145, 0.7072, 0.5715],
        ["Gaussian Naive Bayes", 0.7885, 0.8692, 0.6278, 0.7990, 0.7032, 0.5521],
        ["SVM", 0.8430, 0.9034, 0.7441, 0.7608, 0.7524, 0.6375],
    ],
    columns=[
        "Model", "Accuracy", "AUC",
        "Precision", "Recall", "F1", "MCC"
    ],
)

# Saved model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

# Chart colors
COLORS = [
    "#35d5d0",
    "#9078ff",
    "#ffbe6d",
    "#f184ac",
    "#52a6ff",
    "#b39cff",
]


# ============================================================
# 3. CUSTOM DARK FRONTEND
# ============================================================

st.html("""
<style>

.stApp {
    background: #0b1020;
    color: #edf4ff;
}

[data-testid="stSidebar"] {
    background: #121d34;
    border-right: 1px solid #2d4261;
}

[data-testid="stSidebar"] * {
    color: #e8f1ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f4f8ff !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(130deg, #192849, #202342);
    border: 1px solid #384a77;
    border-radius: 16px;
    padding: 18px 15px;
    min-height: 125px;
}

[data-testid="stMetricLabel"] {
    color: #a5d9ed;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 800;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #5876a8;
    border-radius: 16px;
    padding: 10px;
}

[data-testid="stTabs"] button {
    color: #d6e7fa;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #47e3db;
}

div.stDownloadButton > button {
    background: #197d93;
    color: white;
    border-radius: 12px;
    border: 1px solid #45cbd0;
}

.hero-box {
    background: linear-gradient(
        110deg,
        #312070,
        #174e9c 60%,
        #087e89
    );
    border: 1px solid #5978bf;
    border-radius: 22px;
    padding: 32px 35px;
}

.hero-kicker {
    color: #8bfff0;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}

.hero-title {
    color: white;
    font-size: clamp(31px, 4vw, 50px);
    font-weight: 850;
    margin: 12px 0;
}

.hero-copy {
    color: #e5efff;
    font-size: 16px;
    max-width: 700px;
}


/* ==========================================
   FIX STREAMLIT TOP HEADER
========================================== */

[data-testid="stHeader"] {
    background: #0b1020 !important;
}

[data-testid="stToolbar"] {
    background: #0b1020 !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* ==========================================
   FIX SIDEBAR DROPDOWN COLORS
========================================== */

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1c2b47 !important;
    border: 1px solid #536a91 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #75e7e2 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    color: #ffffff !important;
}

/* ==========================================
   FIX CSV UPLOAD BOX
========================================== */

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #192a45 !important;
    border: 1px dashed #5e92cf !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #213452 !important;
    border: 1px dashed #6686b4 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #334f77 !important;
    color: white !important;
    border: 1px solid #7198c8 !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #e2ecff !important;
}

/* ==========================================
   FIX DROPDOWN POPUP
========================================== */

[data-baseweb="popover"] {
    background: #1c2b47 !important;
}

[data-baseweb="popover"] [role="listbox"] {
    background: #1c2b47 !important;
    color: #ffffff !important;
}

[data-baseweb="popover"] [role="option"] {
    color: #ffffff !important;
}

</style>
""")


# ============================================================
# 4. SIDEBAR WITH INSURANCE IMAGE
# ============================================================
# Create the sidebar
with st.sidebar:

    # ------------------------------------------
    # SIDEBAR INSURANCE IMAGE
    # ------------------------------------------

    # Get the image location
    sidebar_image = ASSETS / "sidebar_insurance.png"

    # Display image only if it exists
    if sidebar_image.is_file():

        st.image(
            str(sidebar_image),
            use_container_width=True
        )

    # ------------------------------------------
    # SIDEBAR HEADING
    # ------------------------------------------

    st.title("🛡️ ClaimLens")

    st.caption("INSURANCE ANALYTICS CONTROL PANEL")

    st.divider()

    # ------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------

    # Select one of the five trained models
    selected_model = st.selectbox(
        "Choose prediction model",
        list(MODEL_FILES.keys()),
        key="claimlens_model_selector"
    )

    # ------------------------------------------
    # CSV FILE UPLOAD
    # ------------------------------------------

    # Only ONE file uploader is needed
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        key="claimlens_csv_upload"
    )

    st.divider()

    # ------------------------------------------
    # CLAIM CLASS INFORMATION
    # ------------------------------------------

    st.markdown("### 🛡️ Claim Categories")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✓ No Claim")

    with col2:
        st.warning("⚠ Claim")

    st.divider()
    # ------------------------------------------
    # DISPLAY ALL FIVE MODELS
    # ------------------------------------------

    st.markdown("### 🧠 Available Models")

    # Use the existing model dictionary
    model_names = list(MODEL_FILES.keys())

    # Display all five models
    for index, name in enumerate(model_names, start=1):

        # Check whether this is the selected model
        if name == selected_model:

            # Highlight the currently selected model
            st.markdown(
                f"🟢 **{index}. {name}**"
            )

        else:

            # Display other models normally
            st.markdown(
                f"⚪ {index}. {name}"
            )


    # ------------------------------------------
    # SIDEBAR FOOTER
    # ------------------------------------------

    st.divider()

    st.caption("🚘 ClaimLens Studio")
    st.caption("Smart Insurance Analytics")


# ============================================================
# 5. MAIN HERO BANNER WITH CAR IMAGE
# ============================================================

left_hero, right_hero = st.columns(
    [2.2, 1],
    gap="large"
)

with left_hero:

    # st.html renders HTML without showing raw HTML tags
    st.html("""
    <div class="hero-box">

        <div class="hero-kicker">
            PREDICTIVE INSURANCE INTELLIGENCE
        </div>

        <div class="hero-title">
            🚘 ClaimLens Studio
        </div>

        <div class="hero-copy">
            Understand insurance claim predictions.
            Compare models, explore performance,
            and uncover insights from your data.
        </div>

    </div>
    """)

with right_hero:

    hero_image = ASSETS / "hero_car.png"

    if hero_image.is_file():

        st.image(
            str(hero_image),
            use_container_width=True
        )

    else:

        st.info(
            "🛡️ Smart insights for safer insurance decisions"
        )

st.write("")


# ============================================================
# 6. LOAD SAVED MODELS AND PREPROCESSING
# ============================================================

@st.cache_resource
def load_artifacts():

    # Load training-fitted preprocessing objects
    preprocessing = {

        "imputer": joblib.load(
            MODELS / "imputer.joblib"
        ),

        "postal_encoder": joblib.load(
            MODELS / "postal_encoder.joblib"
        ),

        "scaler": joblib.load(
            MODELS / "scaler.joblib"
        ),

        "feature_columns": joblib.load(
            MODELS / "feature_columns.joblib"
        ),
    }

    # Load each saved classifier
    trained_models = {}

    for name, filename in MODEL_FILES.items():

        trained_models[name] = joblib.load(
            MODELS / filename
        )

    return preprocessing, trained_models


# ============================================================
# 7. CATEGORY MAPPINGS USED DURING TRAINING
# ============================================================

MAPPINGS = {

    "AGE": {
        "16-25": 0,
        "26-39": 1,
        "40-64": 2,
        "65+": 3
    },

    "DRIVING_EXPERIENCE": {
        "0-9y": 0,
        "10-19y": 1,
        "20-29y": 2,
        "30y+": 3
    },

    "EDUCATION": {
        "none": 0,
        "high school": 1,
        "university": 2
    },

    "INCOME": {
        "poverty": 0,
        "working class": 1,
        "middle class": 2,
        "upper class": 3
    },

    "GENDER": {
        "female": 0,
        "male": 1
    },

    "RACE": {
        "majority": 0,
        "minority": 1
    },

    "VEHICLE_YEAR": {
        "before 2015": 0,
        "after 2015": 1
    },

    "VEHICLE_TYPE": {
        "sedan": 0,
        "sports car": 1
    }
}


# ============================================================
# 8. PREPROCESS UPLOADED TEST DATA
# ============================================================

def preprocess_uploaded_data(raw_features, prep):

    # Work on a copy of the uploaded features
    X = raw_features.copy()

    # Columns where missing values were handled
    numeric_missing = [
        "CREDIT_SCORE",
        "ANNUAL_MILEAGE"
    ]

    # Use the saved training median imputer
    X[numeric_missing] = prep["imputer"].transform(
        X[numeric_missing]
    )

    # Apply training category mappings
    for column, mapping in MAPPINGS.items():

        X[column] = X[column].map(mapping)

    # Encode postal codes using the saved encoder
    postal_array = prep["postal_encoder"].transform(
        X[["POSTAL_CODE"]]
    )

    # Obtain names for encoded postal code columns
    postal_names = prep[
        "postal_encoder"
    ].get_feature_names_out(
        ["POSTAL_CODE"]
    )

    # Create encoded postal code DataFrame
    postal_df = pd.DataFrame(
        postal_array,
        columns=postal_names,
        index=X.index
    )

    # Combine encoded features
    X = pd.concat(
        [
            X.drop(columns=["POSTAL_CODE"]),
            postal_df
        ],
        axis=1
    )

    # Retrieve exact training feature order
    expected_columns = prep["feature_columns"]

    # Check for missing columns
    missing = set(expected_columns) - set(X.columns)

    if missing:

        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    # Arrange columns in training order
    X = X[expected_columns]

    # Check for invalid values
    if X.isna().any().any():

        raise ValueError(
            "Some values are missing or contain unknown categories."
        )

    return X


# ============================================================
# 9. MODEL COMPARISON BAR CHART
# ============================================================

def render_reference_chart():

    # Short names for the chart
    model_names = [
        "Logistic\nRegression",
        "Decision\nTree",
        "KNN",
        "Naive\nBayes",
        "SVM"
    ]

    # Positions of the models
    x = np.arange(len(model_names))

    # Create the figure
    fig, ax = plt.subplots(
        figsize=(10, 4.6),
        dpi=120
    )

    # Apply dark colors
    fig.patch.set_facecolor("#101a30")
    ax.set_facecolor("#101a30")

    # Display three important performance metrics
    chart_metrics = [
        ("Accuracy", "#50a9fa"),
        ("AUC", "#a787ff"),
        ("F1", "#42d9c1")
    ]

    for j, (metric, color) in enumerate(chart_metrics):

        ax.bar(
            x + (j - 1) * 0.23,
            REFERENCE[metric],
            width=0.22,
            label=metric,
            color=color
        )

    # Show model names
    ax.set_xticks(
        x,
        model_names,
        color="#edf4ff"
    )

    # Set score range
    ax.set_ylim(0, 1.05)

    # Format chart axis
    ax.tick_params(
        axis="y",
        colors="#cedbf2"
    )

    ax.grid(
        axis="y",
        alpha=0.18
    )

    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        facecolor="#18263e",
        edgecolor="#445776",
        labelcolor="white",
        ncol=3
    )

    # Remove unnecessary borders
    for spine in ax.spines.values():

        spine.set_visible(False)

    fig.tight_layout()

    # Display graph on the website
    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 10. SIX-METRIC PERFORMANCE GRAPH
# ============================================================

def render_six_metric_chart(scores):

    # Obtain metric names and values
    names = list(scores)

    values = [
        scores[name] for name in names
    ]

    # Create figure
    fig, ax = plt.subplots(
        figsize=(9, 4.3),
        dpi=120
    )

    fig.patch.set_facecolor("#101a30")
    ax.set_facecolor("#101a30")

    # Draw one bar for each metric
    for i, (name, value) in enumerate(
        zip(names, values)
    ):

        if value is not None:

            ax.barh(
                i,
                value,
                color=COLORS[i],
                height=0.6
            )

            # Display numeric value after the bar
            ax.text(
                value + 0.015,
                i,
                f"{value:.4f}",
                color="white",
                va="center",
                fontsize=10
            )

    # Display metric names
    ax.set_yticks(
        range(len(names)),
        names,
        color="#edf4ff"
    )

    # Put Accuracy at the top
    ax.invert_yaxis()

    # Most metrics range from zero to one
    # Negative MCC values remain visible
    ax.set_xlim(-0.05, 1.13)

    ax.tick_params(
        axis="x",
        colors="#cedbf2"
    )

    ax.grid(
        axis="x",
        alpha=0.15
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():

        spine.set_visible(False)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)



# ============================================================
# 11. FIVE-MODEL PERFORMANCE COMPARISON FUNCTION
# ============================================================

# Create a reusable function to display all five models
def show_model_comparison():

    # Display section heading
    st.subheader("🏆 Five-Model Performance Comparison")

    # Create tabs for table and graph
    comparison_tab, chart_tab = st.tabs([
        "📋 Comparison Table",
        "📊 Comparison Chart"
    ])

    # Show performance table
    with comparison_tab:

        # Select numeric metric columns
        metric_columns = [
            column
            for column in REFERENCE.columns
            if column != "Model"
        ]

        # Display formatted comparison table
        st.dataframe(
            REFERENCE.style.format({
                column: "{:.4f}"
                for column in metric_columns
            }),
            hide_index=True,
            use_container_width=True
        )

    # Show performance chart
    with chart_tab:

        render_reference_chart()



# ============================================================
# 12. EVALUATE UPLOADED DATA
# ============================================================

# Check whether a CSV file has been uploaded
if uploaded_file is None:

    # Display the comparison of all five models
    show_model_comparison()

    # Stop execution until a CSV is uploaded
    st.stop()

else:

    try:

        # ------------------------------------------
        # LOAD PRE-TRAINED MODELS
        # ------------------------------------------

        # Load saved preprocessing objects and classifiers
        prep, trained_models = load_artifacts()

        # ------------------------------------------
        # READ UPLOADED CSV
        # ------------------------------------------

        # Read uploaded CSV into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Check whether the dataset is empty
        if df.empty:
            raise ValueError(
                "The uploaded CSV is empty."
            )

        # Check whether OUTCOME exists
        if "OUTCOME" not in df.columns:
            raise ValueError(
                "The uploaded CSV must contain an OUTCOME column."
            )

        # Validate binary target values
        if (
            df["OUTCOME"].isna().any()
            or not df["OUTCOME"].isin([0, 1]).all()
        ):
            raise ValueError(
                "OUTCOME must contain only 0 and 1."
            )

        # ------------------------------------------
        # SEPARATE FEATURES AND TARGET
        # ------------------------------------------

        # Store actual target values
        y_true = df["OUTCOME"].astype(int)

        # Remove ID and target columns
        X_raw = df.drop(
            columns=["ID", "OUTCOME"],
            errors="ignore"
        )

        # ------------------------------------------
        # PREPROCESS INPUT FEATURES
        # ------------------------------------------

        # Apply the same preprocessing used in training
        X_encoded = preprocess_uploaded_data(
            X_raw,
            prep
        )

        # Logistic Regression and SVM require scaled data
        if selected_model in [
            "Logistic Regression",
            "SVM"
        ]:

            # Use the saved training scaler
            X_model = pd.DataFrame(
                prep["scaler"].transform(X_encoded),
                columns=prep["feature_columns"],
                index=X_encoded.index
            )

        else:

            # Decision Tree and Naive Bayes use encoded data
            # The saved KNN pipeline includes its own scaler
            X_model = X_encoded

        # ------------------------------------------
        # MAKE PREDICTIONS
        # ------------------------------------------

        # Select the previously trained classifier
        model = trained_models[selected_model]

        # Predict insurance claim classes
        y_pred = model.predict(
            X_model
        ).astype(int)

        # Predict class 1 probabilities
        y_score = model.predict_proba(
            X_model
        )[:, 1]

        # ------------------------------------------
        # CALCULATE SIX EVALUATION METRICS
        # ------------------------------------------

        scores = {

            # Accuracy
            "Accuracy": accuracy_score(
                y_true,
                y_pred
            ),

            # Area Under ROC Curve
            "AUC": (
                roc_auc_score(y_true, y_score)
                if y_true.nunique() == 2
                else None
            ),

            # Precision
            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            # Recall
            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            # F1 Score
            "F1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            # Matthews Correlation Coefficient
            "MCC": matthews_corrcoef(
                y_true,
                y_pred
            )
        }

        # ------------------------------------------
        # CALCULATE CONFUSION MATRIX
        # ------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

    # ------------------------------------------
    # HANDLE ERRORS
    # ------------------------------------------

    except Exception as error:

        # Display the error message
        st.error(
            f"Cannot evaluate this file: {error}"
        )

        # Stop further execution
        st.stop()


    # ========================================================
    # 13. SIX PERFORMANCE METRIC CARDS
    # ========================================================

    st.divider()

    st.subheader(
        f"📈 Active Model Performance — {selected_model}"
    )

    st.caption(
        f"{len(df):,} records evaluated using a saved "
        "model. No retraining performed."
    )

    # Display three cards in each row
    metric_names = list(scores)

    for start in [0, 3]:

        columns = st.columns(3)

        for column, name in zip(
            columns,
            metric_names[start:start + 3]
        ):

            value = scores[name]

            column.metric(
                name,
                "N/A" if value is None else f"{value:.4f}"
            )

    # Count correct predictions
    correct_predictions = int(
        (y_true.to_numpy() == y_pred).sum()
    )

    st.success(
        f"{correct_predictions:,} out of {len(df):,} "
        "predictions are correct."
    )


    # ========================================================
    # 14. PERFORMANCE GRAPH FOR ALL SIX METRICS
    # ========================================================

    st.subheader(
        "📊 Six-Metric Performance Profile"
    )

    render_six_metric_chart(scores)


    # ========================================================
    # 15. MODEL EVALUATION TABS
    # ========================================================

    tab_cm, tab_report, tab_predictions = st.tabs([
        "🧩 Confusion Matrix",
        "📑 Classification Report",
        "🔍 Predictions & Download"
    ])


    # ========================================================
    # 16. CONFUSION MATRIX
    # ========================================================

    with tab_cm:

        col_plot, col_numbers = st.columns(
            [1.7, 1]
        )

        with col_plot:

            # Create a confusion matrix plot
            fig, ax = plt.subplots(
                figsize=(5.5, 4.2),
                dpi=120
            )

            fig.patch.set_facecolor("#101a30")
            ax.set_facecolor("#101a30")

            # Show the matrix with color shading
            ax.imshow(
                cm,
                cmap="PuBu",
                alpha=0.95
            )

            ax.set_xticks(
                [0, 1],
                ["No Claim", "Claim"],
                color="white"
            )

            ax.set_yticks(
                [0, 1],
                ["No Claim", "Claim"],
                color="white"
            )

            ax.set_xlabel(
                "Predicted",
                color="white"
            )

            ax.set_ylabel(
                "Actual",
                color="white"
            )

            # Write counts inside the matrix cells
            for i in range(2):

                for j in range(2):

                    text_color = (
                        "#131d36"
                        if cm[i, j] < cm.max() * 0.55
                        else "white"
                    )

                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:,}",
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color=text_color
                    )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        # Show confusion matrix summary
        with col_numbers:

            st.metric(
                "Correct Claims",
                int(cm[1, 1])
            )

            st.metric(
                "Missed Claims",
                int(cm[1, 0])
            )

            st.metric(
                "False Alarms",
                int(cm[0, 1])
            )

            st.metric(
                "Correct No-Claims",
                int(cm[0, 0])
            )


    # ========================================================
    # 17. CLASSIFICATION REPORT
    # ========================================================

    with tab_report:

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        # Convert the report into a table
        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )


    # ========================================================
    # 18. PREDICTIONS AND CSV DOWNLOAD
    # ========================================================

    with tab_predictions:

        # Copy the uploaded dataset
        output = df.copy()

        # Add predicted labels
        output["PREDICTED_OUTCOME"] = y_pred

        # Add prediction probabilities
        output["CLAIM_PROBABILITY"] = np.round(
            y_score,
            4
        )

        # Indicate whether each prediction is correct
        output["CORRECT_PREDICTION"] = (
            y_true.to_numpy() == y_pred
        )

        # Preview first 100 records
        st.dataframe(
            output.head(100),
            use_container_width=True
        )

        st.caption(
            "Showing 100 records. "
            "The downloadable file includes every row."
        )

                # Download predictions
        st.download_button(
            label="⬇ Download Prediction Results",
            data=output.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="claimlens_predictions.csv",
            mime="text/csv"
        )


# ============================================================
# FINAL SECTION: FIVE-MODEL PERFORMANCE COMPARISON
# ============================================================

# Add a divider before the final comparison section
st.divider()

# Show comparison at the bottom of the page
show_model_comparison()

# ============================================================
# 19. DECORATIVE FOOTER IMAGE
# ============================================================

st.write("")

footer_image = ASSETS / "road_footer.png"

if footer_image.is_file():

    st.image(
        str(footer_image),
        use_container_width=True
    )

st.caption(
    "ClaimLens Studio · Insurance Analytics"
)
