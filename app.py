import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Product Quality Analysis",
    page_icon="📊",
    layout="wide"
)

# Required columns for validation
REQUIRED_COLUMNS = [
    'id', 'product_name', 'brand', 'amount_lt', 'gm1a_current', 
    'return_rate_lifetime', 'reclaim_rate_lifetime', 'rating_lifetime', 
    'orders_last_365', 'gross_sales_last_365'
]

# Default visible columns for results table
DEFAULT_COLUMNS = [
    'id', 'product_name', 'brand', 'new_category_1', 'amount_lt',
    'return_rate_lifetime', 'reclaim_rate_lifetime', 'rating_lifetime',
    'orders_last_365', 'gross_sales_last_365', 'return_impact', 
    'reclaim_impact', 'total_business_impact'
]

def validate_data(df):
    """Validate that required columns are present in the dataset"""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

def calculate_business_impact(df):
    """Calculate return impact, reclaim impact, and total business impact"""
    df = df.copy()
    
    # Handle missing values
    df['gm1a_current'] = df['gm1a_current'].fillna(0)
    df['return_rate_lifetime'] = df['return_rate_lifetime'].fillna(0)
    df['reclaim_rate_lifetime'] = df['reclaim_rate_lifetime'].fillna(0)
    
    # Calculate impacts
    df['return_impact'] = df['amount_lt'] * df['gm1a_current'] * (df['return_rate_lifetime'] / 100)
    df['reclaim_impact'] = df['amount_lt'] * df['gm1a_current'] * (df['reclaim_rate_lifetime'] / 100)
    df['total_business_impact'] = df['return_impact'] + df['reclaim_impact']
    
    return df

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    filtered_df = df.copy()

    # Apply sales range filters
    if filters['use_min_sales']:
        filtered_df = filtered_df[filtered_df['amount_lt'] >= filters['min_sales']]
    
    if filters['use_max_sales']:
        filtered_df = filtered_df[filtered_df['amount_lt'] <= filters['max_sales']]
    
    conditions = []
    
    # Apply return rate filter if active
    if filters['apply_return_filter']:
        conditions.append(filtered_df['return_rate_lifetime'] >= filters['return_rate'])
    
    # Apply reclaim rate filter if active
    if filters['apply_reclaim_filter']:
        conditions.append(filtered_df['reclaim_rate_lifetime'] >= filters['reclaim_rate'])
    
    # Apply rating filter if active
    if filters['apply_rating_filter']:
        conditions.append(filtered_df['rating_lifetime'] <= filters['max_rating'])
    
    # Combine conditions based on logic
    if conditions:
        if filters['filter_logic'] == "Meet ALL active criteria":
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
        else:  # Meet ANY active criteria
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
        
        filtered_df = filtered_df[combined_condition]
    
    return filtered_df

def create_overview_charts(df):
    """Create visualizations for the overview page"""
    
    # 1. Return rate distribution
    fig_return = px.histogram(
        df, x='return_rate_lifetime', 
        title='Distribution of Return Rates',
        labels={'return_rate_lifetime': 'Return Rate (%)', 'count': 'Number of Products'},
        nbins=30
    )
    
    # 2. Reclaim rate distribution
    fig_reclaim = px.histogram(
        df, x='reclaim_rate_lifetime', 
        title='Distribution of Reclaim Rates',
        labels={'reclaim_rate_lifetime': 'Reclaim Rate (%)', 'count': 'Number of Products'},
        nbins=30
    )
    
    # 3. Rating distribution
    fig_rating = px.histogram(
        df, x='rating_lifetime', 
        title='Distribution of Product Ratings',
        labels={'rating_lifetime': 'Rating', 'count': 'Number of Products'},
        nbins=20
    )
    
    return fig_return, fig_reclaim, fig_rating

def create_sales_distribution_chart(df, thresholds):
    """Create sales distribution chart with user-specified thresholds"""
    fig = px.histogram(
        df, x='amount_lt', 
        title='Distribution of Products by Lifetime Sales',
        labels={'amount_lt': 'Lifetime Sales Amount', 'count': 'Number of Products'},
        nbins=50
    )
    
    # Add vertical lines for thresholds
    for threshold in thresholds:
        fig.add_vline(
            x=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {threshold}"
        )
    
    return fig

def create_scatter_plot(df):
    """Create scatter plot of return rate vs reclaim rate"""
    fig = px.scatter(
        df, 
        x='return_rate_lifetime', 
        y='reclaim_rate_lifetime',
        color='total_business_impact',
        hover_data=['product_name', 'brand'],
        title='Return Rate vs Reclaim Rate (colored by Business Impact)',
        labels={
            'return_rate_lifetime': 'Return Rate (%)',
            'reclaim_rate_lifetime': 'Reclaim Rate (%)',
            'total_business_impact': 'Business Impact'
        }
    )
    return fig

# Main app
def main():
    st.title("🔍 Product Quality Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your product dataset (CSV)", type=['csv'])
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("### Expected dataset structure:")
        st.markdown("The dataset should contain product information with columns for returns, reclaims, ratings, and sales data.")
        return
    
    # Load and validate data
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded successfully! ({len(df)} products). Conditions applied: is_sellable=1, is_active_portfolio=1, excluding categories: Services, Samples, Bonami kitchen, Gift vouchers.")
        
        # Validate required columns
        is_valid, missing_cols = validate_data(df)
        if not is_valid:
            st.error(f"❌ Missing required columns: {missing_cols}")
            return
        
        # Calculate business impact
        df = calculate_business_impact(df)
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Select Page:", ["Overview", "Quality Analysis"])
    
    if page == "Overview":
        show_overview_page(df)
    else:
        show_analysis_page(df)

def show_overview_page(df):
    """Display the overview page with dataset summary and charts"""
    st.header("📈 Dataset Overview")
    
    # Dataset summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    
    with col2:
        active_products = len(df[df.get('is_active', True) == True]) if 'is_active' in df.columns else "N/A"
        st.metric("Active Products", active_products)
    
    with col3:
        avg_return_rate = df['return_rate_lifetime'].mean()
        st.metric("Avg Return Rate", f"{avg_return_rate:.1f}%")
    
    with col4:
        avg_reclaim_rate = df['reclaim_rate_lifetime'].mean()
        st.metric("Avg Reclaim Rate", f"{avg_reclaim_rate:.1f}%")
    
    st.markdown("---")
    
    # Charts section
    st.subheader("📊 Data Visualizations")
    
    # Distribution charts
    fig_return, fig_reclaim, fig_rating = create_overview_charts(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_return, use_container_width=True)
    with col2:
        st.plotly_chart(fig_reclaim, use_container_width=True)
    
    st.plotly_chart(fig_rating, use_container_width=True)
    
    # Sales distribution with thresholds
    st.subheader("💰 Sales Distribution Analysis")
    threshold_input = st.text_input(
        "Enter sales thresholds (comma-separated) and below you can see the histogram of sales + further below you can see exact number of products above each threshold:", 
        value="5,10,20,50",
        help="Enter threshold values separated by commas to highlight on the chart"
    )
    
    try:
        thresholds = [float(x.strip()) for x in threshold_input.split(',') if x.strip()]
        fig_sales = create_sales_distribution_chart(df, thresholds)
        st.plotly_chart(fig_sales, use_container_width=True)
        
        # Show counts above thresholds
        st.markdown("##### 📊 Products Above Sales Thresholds")
        threshold_cols = st.columns(len(thresholds))
        for i, threshold in enumerate(thresholds):
            count = len(df[df['amount_lt'] >= threshold])
            percentage = (count / len(df)) * 100
            with threshold_cols[i]:
                st.metric(
                    f"≥ {threshold} sales", 
                    f"{count} products",
                    f"{percentage:.1f}%"
                )
    except:
        st.error("Please enter valid numeric thresholds separated by commas")
    
    # Scatter plot
    st.subheader("🎯 Return vs Reclaim Analysis")
    fig_scatter = create_scatter_plot(df)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top products by business impact
    st.subheader("⚠️ Top Products by Business Impact")
    top_n = st.number_input("Number of products to display:", min_value=5, max_value=50, value=10)
    
    top_products = df.nlargest(top_n, 'total_business_impact')[
        ['product_name', 'brand', 'return_rate_lifetime', 'reclaim_rate_lifetime', 
         'total_business_impact']
    ].round(2)
    
    st.dataframe(top_products, use_container_width=True)

def show_analysis_page(df):
    """Display the quality analysis page with filtering capabilities"""
    st.header("🎯 Quality Analysis Tool")
    st.markdown("Use the filters below to identify problematic products based on your criteria.")
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Filters")

    use_min_sales = st.sidebar.checkbox("Set Minimum Sales", value=True)
    min_sales = st.sidebar.number_input(
        "Minimum Sales for Analysis:", 
        min_value=0, 
        value=10,
        disabled=not use_min_sales,
        help="Only analyze products with at least this many lifetime sales"
    )

    use_max_sales = st.sidebar.checkbox("Set Maximum Sales")
    max_sales = st.sidebar.number_input(
        "Maximum Sales for Analysis:", 
        min_value=0, 
        value=1000,
        disabled=not use_max_sales,
        help="Only analyze products with at most this many lifetime sales"
    )
    
    st.sidebar.markdown("### Active Filters")
    
    # Return rate filter
    apply_return_filter = st.sidebar.checkbox("Apply Return Rate Filter")
    return_rate = st.sidebar.number_input(
        "Return Rate Threshold (%):", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0,
        disabled=not apply_return_filter
    )
    
    # Reclaim rate filter
    apply_reclaim_filter = st.sidebar.checkbox("Apply Reclaim Rate Filter")
    reclaim_rate = st.sidebar.number_input(
        "Reclaim Rate Threshold (%):", 
        min_value=0.0, 
        max_value=100.0, 
        value=15.0,
        disabled=not apply_reclaim_filter
    )
    
    # Rating filter
    apply_rating_filter = st.sidebar.checkbox("Apply Rating Filter (maximum)")
    max_rating = st.sidebar.number_input(
        "Maximum Rating:", 
        min_value=0.0, 
        max_value=10.0, 
        value=3.0,
        disabled=not apply_rating_filter
    )
    
    # Filter logic
    filter_logic = st.sidebar.selectbox(
        "Filter Logic:", 
        ["Meet ALL active criteria", "Meet ANY active criteria"]
    )
    
    # Apply filters
    filters = {
        'use_min_sales': use_min_sales,
        'min_sales': min_sales,
        'use_max_sales': use_max_sales,
        'max_sales': max_sales,
        'apply_return_filter': apply_return_filter,
        'return_rate': return_rate,
        'apply_reclaim_filter': apply_reclaim_filter,
        'reclaim_rate': reclaim_rate,
        'apply_rating_filter': apply_rating_filter,
        'max_rating': max_rating,
        'filter_logic': filter_logic
    }
    
    filtered_df = apply_filters(df, filters)
    
    # Results summary
    st.subheader("📊 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products Found", len(filtered_df))
    
    with col2:
        total_return_impact = filtered_df['return_impact'].sum()
        st.metric("Total Return Impact", f"{total_return_impact:,.0f}")
    
    with col3:
        total_reclaim_impact = filtered_df['reclaim_impact'].sum()
        st.metric("Total Reclaim Impact", f"{total_reclaim_impact:,.0f}")
    
    with col4:
        total_business_impact = filtered_df['total_business_impact'].sum()
        st.metric("Total Business Impact", f"{total_business_impact:,.0f}")
    
    # Results table
    if len(filtered_df) > 0:
        st.subheader("📋 Problematic Products")
        
        # Column selection
        st.markdown("**Customize table columns:**")
        available_columns = [col for col in df.columns if col not in ['return_impact', 'reclaim_impact', 'total_business_impact']]
        available_columns.extend(['return_impact', 'reclaim_impact', 'total_business_impact'])
        
        selected_columns = st.multiselect(
            "Select additional columns to display:",
            available_columns,
            default=[col for col in DEFAULT_COLUMNS if col in df.columns],
            help="Choose which columns you want to see in the results table"
        )
        
        if selected_columns:
            # Display filtered results
            display_df = filtered_df[selected_columns].round(2)
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"problematic_products_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one column to display")
    else:
        st.info("No products match the selected criteria. Try adjusting your filters.")

if __name__ == "__main__":
    main()