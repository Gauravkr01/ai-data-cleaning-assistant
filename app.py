import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
# ✅ FIX 1: Never hardcode API keys. Use st.secrets or environment variables.
# In .streamlit/secrets.toml add: GROQ_API_KEY = "your_key"
# OR set the environment variable GROQ_API_KEY before running.
from groq import Groq
import os

# Get API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize client
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# -----------------------------------------------
# AI EXPLANATION
# -----------------------------------------------
def get_ai_explanation(prompt: str) -> str:
    """Call Groq LLM; return explanation string or error message."""
    if client is None:
        return "⚠️ Groq API key not configured. Add GROQ_API_KEY to .streamlit/secrets.toml"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,      # ✅ FIX 2: Prevent runaway token cost
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling AI: {e}"


# -----------------------------------------------
# SAFE DISPLAY HELPER
# -----------------------------------------------
def safe_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ FIX 3: Properly handle NaN and object columns for PyArrow/st.dataframe.
    Converts object columns to str, replaces actual NaN (not just the string 'nan').
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace({float("nan"): "", "nan": "", "None": ""})
    return df


# -----------------------------------------------
# CLEANING FUNCTIONS
# -----------------------------------------------
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect and parse date-like object columns."""
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        converted = pd.to_datetime(sample, errors="coerce")
        if converted.notna().sum() > len(sample) * 0.7:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def remove_duplicates(df: pd.DataFrame):
    """Returns (cleaned_df, num_found, num_removed)."""
    before = len(df)
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, n_dupes, removed


def smart_convert_column(col: pd.Series) -> pd.Series:
    """Try to extract and convert a column to numeric."""
    extracted = col.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.isna().all():
        return col
    if numeric.notna().sum() > len(col) * 0.6:
        return numeric
    return col


def smart_convert_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply smart numeric conversion to all non-datetime object columns."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if df[col].dtype == "object":
            df[col] = smart_convert_column(df[col])
    return df


def fill_numeric(df: pd.DataFrame):
    """Fill numeric NaNs with mean (low skew) or median (high skew). Returns (df, report)."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    report = []
    for col in num_cols:
        missing_before = df[col].isnull().sum()
        if missing_before == 0:
            continue
        skewness = df[col].skew()
        fill_val = df[col].median() if abs(skewness) > 1 else df[col].mean()
        df[col] = df[col].fillna(fill_val)
        filled = missing_before - df[col].isnull().sum()
        report.append(f"'{col}': filled {filled} values with {'median' if abs(skewness) > 1 else 'mean'}")
    return df, report


def fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Fill categorical NaNs with mode."""
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase all object columns. Must run AFTER numeric conversion."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
    return df


# -----------------------------------------------
# UI STYLING
# -----------------------------------------------
st.set_page_config(page_title="AI Data Cleaning Assistant", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4f8 0%, #dce8f5 100%);
}
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 10px;
}
thead tr th { text-align: center !important; }
tbody tr td { text-align: center !important; }
h1, h2, h3 { color: #1e2d3d; }

.card {
    background-color: white;
    padding: 20px 24px;
    border-radius: 14px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.metric-box {
    background: #f0f7ff;
    border-radius: 10px;
    padding: 12px 20px;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; color: #2563eb; }
.metric-box .lbl { font-size: 0.8rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Data Cleaning Assistant")
st.caption("Upload, clean, visualize, and predict — all in one place.")

# -----------------------------------------------
# FILE UPLOAD
# -----------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("👆 Please upload a CSV file to get started.")
    st.stop()

# Reset state when a new file is uploaded
if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
    df_raw = pd.read_csv(uploaded_file)
    st.session_state.df = df_raw
    st.session_state.cleaned = False
    st.session_state.file_name = uploaded_file.name
    st.session_state.cleaning_log = []

df = st.session_state.df

# -----------------------------------------------
# SECTION 1 — DATASET PREVIEW
# -----------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")

col_left, col_right = st.columns([3, 1])
with col_left:
    selected_cols = st.multiselect(
        "Select columns to preview",
        df.columns.tolist(),
        default=list(df.columns[:6]),
    )
with col_right:
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="val">{df.shape[0]}</div><div class="lbl">Rows</div></div>
        <div class="metric-box"><div class="val">{df.shape[1]}</div><div class="lbl">Columns</div></div>
        <div class="metric-box"><div class="val">{df.isnull().sum().sum()}</div><div class="lbl">Missing</div></div>
    </div>
    """, unsafe_allow_html=True)

if selected_cols:
    st.dataframe(safe_display(df[selected_cols].head(10)), use_container_width=True)

with st.expander("📌 Column Data Types"):
    dtype_df = pd.DataFrame({"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)})
    st.dataframe(dtype_df, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 2 — MISSING VALUES
# -----------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Missing Values")

missing = df.isnull().sum()
total_missing = int(missing.sum())

styled_missing = (
    missing.to_frame(name="Missing Count")
    .style.background_gradient(cmap="Reds")
    .set_properties(**{"text-align": "center"})
)
st.table(styled_missing)

if total_missing == 0:
    st.success("🎉 No missing values found!")
else:
    st.warning(f"⚠️ Total missing values: {total_missing}")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 3 — CLEANING OPTIONS
# -----------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧹 Cleaning Options")

option = st.radio(
    "Choose a cleaning method",
    [
        "None",
        "Remove Duplicates",
        "Fix Data Types",
        "Fill Numeric (Mean/Median)",
        "Fill Categorical (Mode)",
        "Drop Rows with Missing Values",
        "Auto Clean (Recommended 🔥)",
    ],
    horizontal=True,
)

if st.button("▶ Apply Cleaning", type="primary"):
    df_temp = df.copy()

    if option == "None":
        st.info("No cleaning option selected.")

    elif option == "Fix Data Types":
        df_temp = parse_dates(df_temp)
        df_temp = smart_convert_df(df_temp)
        st.session_state.cleaning_log.append("Fixed data types (date detection + numeric conversion)")
        st.success("✅ Data types fixed!")

    elif option == "Fill Numeric (Mean/Median)":
        df_temp = smart_convert_df(df_temp)
        df_temp, report = fill_numeric(df_temp)
        if report:
            st.session_state.cleaning_log.extend(report)
            st.success(f"✅ Filled numeric missing values ({len(report)} columns affected)")
        else:
            st.info("No numeric missing values found.")

    elif option == "Fill Categorical (Mode)":
        df_temp = fill_categorical(df_temp)
        st.session_state.cleaning_log.append("Filled categorical missing values with mode")
        st.success("✅ Categorical values filled!")

    elif option == "Drop Rows with Missing Values":
        rows_before = len(df_temp)
        df_temp = df_temp.dropna().reset_index(drop=True)
        dropped = rows_before - len(df_temp)
        st.session_state.cleaning_log.append(f"Dropped {dropped} rows with missing values")
        st.success(f"✅ Dropped {dropped} rows with missing values.")

    elif option == "Remove Duplicates":
        df_temp, n_dupes, removed = remove_duplicates(df_temp)
        if n_dupes > 0:
            st.session_state.cleaning_log.append(f"Removed {removed} duplicate rows")
            st.success(f"✅ Found {n_dupes} duplicates — removed {removed} rows.")
        else:
            st.success("✅ No duplicate rows found.")

    elif option == "Auto Clean (Recommended 🔥)":
        # Correct pipeline order matters:
        # 1. Parse dates (before type conversion)
        # 2. Convert numeric types
        # 3. Remove duplicates
        # 4. Fill numeric NaNs
        # 5. Fill categorical NaNs
        # 6. Clean strings LAST (after numeric conversion is done)
        df_temp = parse_dates(df_temp)
        df_temp = smart_convert_df(df_temp)
        df_temp, n_dupes, removed = remove_duplicates(df_temp)
        if n_dupes > 0:
            st.session_state.cleaning_log.append(f"Removed {removed} duplicate rows")
        df_temp, report = fill_numeric(df_temp)
        if report:
            st.session_state.cleaning_log.extend(report)
        df_temp = fill_categorical(df_temp)
        df_temp = clean_strings(df_temp)
        st.session_state.cleaning_log.append("✅ Auto clean pipeline completed")
        st.success("🔥 Dataset fully cleaned with Auto Clean!")

    if option != "None":
        st.session_state.df = df_temp
        st.session_state.cleaned = True

with st.expander("📋 Cleaning Log"):
    if st.session_state.cleaning_log:
        for entry in st.session_state.cleaning_log:
            st.write("✔️", entry)
    else:
        st.write("No cleaning actions applied yet.")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 4 — POST-CLEAN SUMMARY (only if cleaned)
# -----------------------------------------------
if st.session_state.get("cleaned", False):
    df_clean = st.session_state.df

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Cleaned Dataset Summary")

    missing_after = df_clean.isnull().sum()
    styled_after = (
        missing_after.to_frame(name="Missing Count")
        .style.background_gradient(cmap="Greens")
        .set_properties(**{"text-align": "center"})
    )
    st.table(styled_after)

    remaining = int(missing_after.sum())
    if remaining == 0:
        st.success("🎉 No missing values remain — dataset is fully clean!")
    else:
        st.warning(f"⚠️ {remaining} missing values remain. Try another cleaning method above.")

    # ✅ FIX 4: Use updated df_clean columns for preview (was using stale selected_cols)
    valid_cols = [c for c in selected_cols if c in df_clean.columns]
    if valid_cols:
        st.dataframe(safe_display(df_clean[valid_cols].head(10)), use_container_width=True)

    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------
    # SECTION 5 — VISUALIZATION
    # -------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Data Visualization")

    all_columns = df_clean.columns.tolist()
    numeric_columns = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-axis", all_columns)
    with col2:
        y_axis = st.selectbox("Y-axis (optional)", ["None"] + all_columns)
    with col3:
        chart_type = st.selectbox(
            "Chart Type",
            ["Histogram", "Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Correlation Heatmap"],
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    plt.style.use("ggplot")

    try:
        if chart_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df_clean[x_axis]):
                ax.hist(df_clean[x_axis].dropna(), bins=30, color="#3b82f6", edgecolor="white")
                ax.set_title(f"Histogram of {x_axis}")
                ax.set_xlabel(x_axis)
            else:
                st.warning("⚠️ Histogram requires a numeric column.")

        elif chart_type == "Bar Chart":
            counts = df_clean[x_axis].value_counts().head(10)
            ax.bar(counts.index.astype(str), counts.values, color="#3b82f6")
            ax.set_title(f"Top Categories: {x_axis}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        elif chart_type == "Scatter Plot":
            if y_axis != "None":
                ax.scatter(df_clean[x_axis], df_clean[y_axis], alpha=0.5, color="#3b82f6", edgecolors="none")
                ax.set_title(f"{x_axis} vs {y_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                plt.tight_layout()
            else:
                st.warning("⚠️ Please select a Y-axis for Scatter Plot.")

        elif chart_type == "Line Chart":
            if y_axis != "None":
                ax.plot(df_clean[x_axis], df_clean[y_axis], color="#3b82f6", linewidth=1.5)
                ax.set_title(f"{x_axis} vs {y_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                plt.tight_layout()
            else:
                st.warning("⚠️ Please select a Y-axis for Line Chart.")

        elif chart_type == "Box Plot":
            if pd.api.types.is_numeric_dtype(df_clean[x_axis]):
                ax.boxplot(df_clean[x_axis].dropna(), patch_artist=True,
                           boxprops=dict(facecolor="#bfdbfe"))
                ax.set_title(f"Box Plot: {x_axis}")
                plt.tight_layout()
            else:
                st.warning("⚠️ Box Plot requires a numeric column.")

        elif chart_type == "Correlation Heatmap":
            if len(numeric_columns) < 2:
                st.warning("⚠️ Need at least 2 numeric columns for a heatmap.")
            else:
                import seaborn as sns
                corr = df_clean[numeric_columns].corr()
                fig, ax = plt.subplots(figsize=(max(6, len(numeric_columns)), max(4, len(numeric_columns) - 1)))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
                ax.set_title("Correlation Heatmap")
                plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Chart error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------
    # SECTION 6 — INSIGHTS + AI EXPLANATION
    # -------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Column Insights + AI Explanation")

    col_selected = st.selectbox("Select a column to analyze", all_columns)

    if pd.api.types.is_numeric_dtype(df_clean[col_selected]):
        col_data = df_clean[col_selected].dropna()
        mean_val = col_data.mean()
        median_val = col_data.median()
        std_val = col_data.std()
        skew_val = col_data.skew()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_val:.2f}")
        c2.metric("Median", f"{median_val:.2f}")
        c3.metric("Std Dev", f"{std_val:.2f}")
        c4.metric("Outliers", len(outliers))

        if skew_val > 1:
            st.info("➡️ Positively skewed — a few large values pull the mean up.")
        elif skew_val < -1:
            st.info("➡️ Negatively skewed — a few small values pull the mean down.")
        else:
            st.info("➡️ Approximately symmetric distribution.")

        prompt = f"""
You are a senior Data Scientist. Explain this numeric column's statistics in simple, clear language (3-4 sentences max).

Column: {col_selected}
Mean: {mean_val:.2f}
Median: {median_val:.2f}
Std Dev: {std_val:.2f}
Skewness: {skew_val:.2f}
Outliers: {len(outliers)}

Give actionable data quality insights.
"""
    else:
        top_vals = df_clean[col_selected].value_counts().head(5)
        st.write("**Top 5 Categories:**")
        st.table(top_vals)
        st.info(f"Most common value: **{top_vals.index[0]}** ({top_vals.iloc[0]} occurrences)")

        prompt = f"""
You are a senior Data Scientist. Analyze this categorical column (2-3 sentences).

Column: {col_selected}
Top values:
{top_vals.to_string()}

Highlight any imbalance, patterns, or concerns.
"""

    if st.button("🤖 Explain with AI"):
        with st.spinner("Thinking like a data scientist... 🤔"):
            result = get_ai_explanation(prompt)
        st.success("**AI Insight:**")
        st.write(f"**{result}**")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------
    # SECTION 7 — ML PREDICTION
    # -------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 ML Prediction")

    target = st.selectbox("Select Target Column", df_clean.columns.tolist())

    if st.button("🚀 Train Model", type="primary"):
        df_ml = df_clean.copy()

        # Encode categoricals
        le = LabelEncoder()
        for col in df_ml.select_dtypes(include="object").columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        # Drop datetime columns (not usable directly in sklearn)
        dt_cols = df_ml.select_dtypes(include="datetime64").columns.tolist()
        if dt_cols:
            df_ml = df_ml.drop(columns=dt_cols)
            st.info(f"ℹ️ Dropped datetime columns: {dt_cols}")

        # ✅ FIX 5: Re-check target exists after dropping
        if target not in df_ml.columns:
            st.error(f"Target column '{target}' was dropped (datetime). Please choose another.")
        else:
            X = df_ml.drop(columns=[target])
            y = df_ml[target]

            if len(X) < 10:
                st.error("❌ Not enough data to train a model (need at least 10 rows).")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                is_classification = y.nunique() <= 10 and (
                    y.dtype == "object" or y.nunique() < 20
                )

                if is_classification:
                    # ✅ FIX 6: Added max_iter to prevent convergence failure
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.success(f"✅ Classification — Accuracy: **{acc:.2%}**")

                    # Feature importance (coefficients)
                    if hasattr(model, "coef_"):
                        coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
                        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": coef})
                        feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
                        fig_fi, ax_fi = plt.subplots(figsize=(7, 3))
                        ax_fi.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#3b82f6")
                        ax_fi.set_title("Top Feature Importances")
                        plt.tight_layout()
                        st.pyplot(fig_fi)
                        plt.close(fig_fi)

                else:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)
                    r2 = model.score(X_test, y_test)
                    st.success(f"✅ Regression — RMSE: **{rmse:.2f}** | R²: **{r2:.4f}**")

                    # Feature importance (coefficients)
                    feat_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Coefficient": np.abs(model.coef_)
                    }).sort_values("Coefficient", ascending=False).head(10)

                    fig_fi, ax_fi = plt.subplots(figsize=(7, 3))
                    ax_fi.barh(feat_df["Feature"][::-1], feat_df["Coefficient"][::-1], color="#10b981")
                    ax_fi.set_title("Top Feature Coefficients (abs)")
                    plt.tight_layout()
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)

    st.markdown("</div>", unsafe_allow_html=True)


