# ============================================================
# SHOPPER SPECTRUM
# Customer Segmentation & Product Recommendation System
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

h1,h2,h3{
    color:#0E4C92;
}

[data-testid="stMetricValue"]{
    font-size:28px;
}

.sidebar .sidebar-content{
    background-color:#F8F9FA;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_models():

    kmeans = joblib.load("kmeans_model.pkl")

    scaler = joblib.load("scaler.pkl")

    similarity_df = joblib.load(
        "product_similarity.pkl"
    )

    product_names = joblib.load(
        "product_names.pkl"
    )

    return (
        kmeans,
        scaler,
        similarity_df,
        product_names
    )


try:

    (
        kmeans,
        scaler,
        similarity_df,
        product_names
    ) = load_models()

except Exception as e:

    st.error(f"Error loading models:\n\n{e}")

    st.stop()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("🛒 Shopper Spectrum")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "👥 Customer Segmentation",

        "🛍 Product Recommendation",

        "📊 Analytics",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developed using

• RFM Analysis

• KMeans Clustering

• Item-Based Collaborative Filtering

• Streamlit
"""
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🛒 Shopper Spectrum")

    st.markdown(
    """
    ### Customer Segmentation & Product Recommendation System

    This application provides intelligent customer segmentation
    using **RFM Analysis + KMeans Clustering**
    and product recommendations using
    **Item-Based Collaborative Filtering**.
    """
    )

    st.markdown("---")

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.metric(

            "Customers",

            "4,338"

        )

    with col2:

        st.metric(

            "Products",

            "3,877"

        )

    with col3:

        st.metric(

            "Customer Segments",

            "4"

        )

    with col4:

        st.metric(

            "Recommendation Model",

            "Ready"

        )

    st.markdown("---")

    left,right=st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write("""

This project applies Machine Learning techniques on Online Retail
transaction data to generate meaningful business insights.

### Features

✅ Customer Segmentation

✅ Product Recommendation

✅ RFM Analysis

✅ KMeans Clustering

✅ PCA Visualization

✅ Item-Based Collaborative Filtering

        """)

    with right:

        st.success("""

### Models Loaded Successfully

✔ KMeans

✔ StandardScaler

✔ Recommendation Engine

✔ Product Dictionary

        """)

    st.markdown("---")

    st.subheader("Business Objective")

    st.write("""

The objective of Shopper Spectrum is to help businesses:

- Identify High-Value Customers

- Identify At-Risk Customers

- Improve Customer Retention

- Increase Cross-selling

- Improve Product Discovery

- Support Marketing Decision Making

    """)

    st.markdown("---")

    st.info(
"""
👈 Use the navigation menu on the left to explore the different modules of the application.
"""
)

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write(
    """
Enter customer RFM values to predict the customer segment.
"""
    )

    st.markdown("---")

    col1,col2,col3=st.columns(3)

    with col1:

        recency=st.number_input(

            "Recency (Days)",

            min_value=1,

            value=30

        )

    with col2:

        frequency=st.number_input(

            "Frequency",

            min_value=1,

            value=5

        )

    with col3:

        monetary=st.number_input(

            "Monetary Value",

            min_value=0.0,

            value=1000.0

        )

    st.markdown("")

    if st.button("Predict Customer Segment",use_container_width=True):

        input_data=pd.DataFrame(

            [[recency,frequency,monetary]],

            columns=[

                "Recency",

                "Frequency",

                "Monetary"

            ]

        )

        scaled_input=scaler.transform(input_data)

        prediction=kmeans.predict(scaled_input)[0]

        cluster_labels={

            0:"Occasional",

            1:"At-Risk",

            2:"High-Value",

            3:"Regular"

        }

        segment=cluster_labels[prediction]

        st.success(
            f"Predicted Segment : **{segment}**"
        )

        segment_messages={

            "High-Value":
            "Excellent customer. Prioritize loyalty rewards, VIP memberships and personalized recommendations.",

            "Regular":
            "Consistent customer. Increase basket value using cross-selling and upselling.",

            "Occasional":
            "Moderately active customer. Encourage repeat purchases through targeted promotions.",

            "At-Risk":
            "Customer inactivity detected. Launch retention campaigns and win-back offers."

        }

        st.info(segment_messages[segment])

# ============================================================
# PRODUCT RECOMMENDATION
# ============================================================

elif page == "🛍 Product Recommendation":

    st.title("🛍 Product Recommendation System")

    st.write(
        """
Select a product and discover the top similar products
using Item-Based Collaborative Filtering.
"""
    )

    st.markdown("---")

    selected_product = st.selectbox(

        "Select Product",

        sorted(product_names)

    )

    st.markdown("")

    if st.button("Generate Recommendations", use_container_width=True):

        recommendations = (

            similarity_df[selected_product]

            .sort_values(ascending=False)

            .iloc[1:6]

        )

        recommendation_df = pd.DataFrame({

            "Rank": range(1,6),

            "Recommended Product": recommendations.index,

            "Similarity Score": recommendations.values.round(3)

        })

        st.success("Top 5 Recommended Products")

        st.dataframe(

            recommendation_df,

            use_container_width=True,

            hide_index=True

        )

        st.bar_chart(

            recommendation_df.set_index("Recommended Product")[
                "Similarity Score"
            ]

        )

# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.write(
        """
Interactive visualizations generated from the trained
machine learning models.
"""
    )

    st.markdown("---")

    st.subheader("Customer Segment Distribution")

    cluster_names = {

        0: "Occasional",

        1: "At-Risk",

        2: "High-Value",

        3: "Regular"

    }

    cluster_counts = pd.Series(

        kmeans.labels_

    ).map(cluster_names).value_counts()

    st.bar_chart(cluster_counts)

    st.markdown("---")

    st.subheader("Recommendation Engine Summary")

    col1,col2,col3=st.columns(3)

    with col1:

        st.metric(

            "Products",

            len(product_names)

        )

    with col2:

        st.metric(

            "Similarity Matrix",

            f"{similarity_df.shape[0]} × {similarity_df.shape[1]}"

        )

    with col3:

        st.metric(

            "Recommendation Type",

            "Item-Based CF"

        )

    st.markdown("---")

    st.subheader("Similarity Matrix Preview")

    st.dataframe(

        similarity_df.iloc[:10,:10],

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Recommendation Statistics")

    st.write(f"Total Products : **{len(product_names):,}**")

    st.write(

        f"Matrix Size : **{similarity_df.shape[0]:,} × {similarity_df.shape[1]:,}**"

    )

    st.write(

        f"Maximum Similarity : **{similarity_df.values.max():.2f}**"

    )

    st.write(

        f"Minimum Similarity : **{similarity_df.values.min():.2f}**"

    )

    st.info("""

The recommendation engine computes cosine similarity
between every pair of products.

Products frequently purchased together receive
higher similarity scores.

""")

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ About":

    st.title("ℹ About Shopper Spectrum")

    st.markdown("---")

    st.subheader("Project Description")

    st.write("""

**Shopper Spectrum** is an end-to-end Customer Analytics
application developed using Machine Learning.

The project combines customer segmentation and
product recommendation techniques to assist
businesses in making data-driven marketing decisions.

""")

    st.markdown("---")

    st.subheader("Machine Learning Techniques Used")

    techniques = pd.DataFrame({

        "Technique":[

            "RFM Analysis",

            "StandardScaler",

            "KMeans Clustering",

            "PCA",

            "Item-Based Collaborative Filtering",

            "Cosine Similarity"

        ],

        "Purpose":[

            "Customer Behaviour Analysis",

            "Feature Scaling",

            "Customer Segmentation",

            "Dimensionality Reduction",

            "Product Recommendation",

            "Product Similarity Measurement"

        ]

    })

    st.dataframe(

        techniques,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.subheader("Business Applications")

    st.markdown("""

- 🎯 Customer Segmentation

- 💰 Targeted Marketing Campaigns

- 📈 Customer Retention

- 🛍 Product Recommendation

- 🔄 Cross Selling

- 📦 Inventory Planning

- 💡 Business Intelligence

""")

    st.markdown("---")

    st.subheader("Technology Stack")

    tech1,tech2=st.columns(2)

    with tech1:

        st.success("""

### Python Libraries

- Pandas

- NumPy

- Scikit-Learn

- Streamlit

- Joblib

""")

    with tech2:

        st.success("""

### Machine Learning

- RFM Analysis

- KMeans

- PCA

- Cosine Similarity

- Collaborative Filtering

""")

    st.markdown("---")

    st.subheader("Project Workflow")

    st.code("""

Online Retail Dataset

        ↓

Data Cleaning

        ↓

Exploratory Data Analysis

        ↓

Feature Engineering

        ↓

RFM Analysis

        ↓

StandardScaler

        ↓

KMeans Clustering

        ↓

Customer Segmentation

        ↓

Item-Based Collaborative Filtering

        ↓

Recommendation Engine

        ↓

Streamlit Dashboard

""")

    st.markdown("---")

    st.info("""

Developed as a Machine Learning Project

Shopper Spectrum

Customer Segmentation & Product Recommendation System

""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "© 2026 Shopper Spectrum | Customer Segmentation & Product Recommendation System"
)