from warnings import filters
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
    'orders_last_365', 'gross_sales_last_365', 'return_order_rate', 'reclaim_order_rate', 'value_available_on_stock', 'available_on_stock'
]

# Default visible columns for results table
DEFAULT_COLUMNS = [
    'id', 'product_name', 'brand', 'new_category_1', 'amount_lt',
    'return_rate_lifetime', 'reclaim_rate_lifetime', 'rating_lifetime', 'total_business_impact'
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

    # Apply return order rate filter if active 
    if filters['apply_return_order_filter']: 
        conditions.append(filtered_df['return_order_rate'] >= filters['return_order_rate']) 

    # Apply reclaim order rate filter if active
    if filters['apply_reclaim_order_filter']:
        conditions.append(filtered_df['reclaim_order_rate'] >= filters['reclaim_order_rate'])

    # Apply value available on stock filter if active
    if filters['apply_value_stock_filter']:
        conditions.append(filtered_df['value_available_on_stock'] >= filters['min_value_stock'])

    # Apply available on stock filter if active
    if filters['apply_available_stock_filter']:
        conditions.append(filtered_df['available_on_stock'] >= filters['min_available_stock'])

    # Apply avg return product count filter if active
    if filters['apply_avg_return_count_filter']:
        conditions.append(filtered_df['avg_return_product_count'] <= filters['max_avg_return_count'])

    # Apply avg reclaim product count filter if active
    if filters['apply_avg_reclaim_count_filter']:
        conditions.append(filtered_df['avg_reclaim_product_count'] <= filters['max_avg_reclaim_count'])

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
        st.success(f"✅ Dataset loaded successfully! ({len(df)} products). Conditions applied: (is_sellable=1 OR is_active=1), is_active_portfolio=1, excluding categories: Services, Samples, Bonami kitchen, Gift vouchers.")
        
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
    page = st.sidebar.radio("Select Page:", ["Overview", "Quality Analysis", "Metric Definitions"])
    
    if page == "Overview":
        show_overview_page(df)
    elif page == "Quality Analysis":
        show_analysis_page(df)
    elif page == "Metric Definitions":
        show_metrics_page()

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
        st.metric("Avg Return Rate Lifetime", f"{avg_return_rate:.1f}%")
    
    with col4:
        avg_reclaim_rate = df['reclaim_rate_lifetime'].mean()
        st.metric("Avg Reclaim Rate Lifetime", f"{avg_reclaim_rate:.1f}%")
    
    st.markdown("---")
    
    # Charts section
    st.subheader("📊 Data Visualizations")
    
    # Distribution charts
    fig_return, fig_reclaim, fig_rating = create_overview_charts(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_return, width='stretch')
    with col2:
        st.plotly_chart(fig_reclaim, width='stretch')
    
    st.plotly_chart(fig_rating, width='stretch')
    
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
        st.plotly_chart(fig_sales, width='stretch')
        
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
    
    # Top products by business impact
    st.subheader("⚠️ Top Products by Business Impact")
    top_n = st.number_input("Number of products to display:", min_value=5, max_value=50, value=10)
    
    top_products = df.nlargest(top_n, 'total_business_impact')[
        ['product_name', 'brand', 'return_rate_lifetime', 'reclaim_rate_lifetime', 'return_impact', 'reclaim_impact',
         'total_business_impact']
    ].round(2)
    
    st.dataframe(top_products, width='stretch')

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
        "Minimum Return Rate Threshold (%):", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0,
        disabled=not apply_return_filter
    )
    
    # Reclaim rate filter
    apply_reclaim_filter = st.sidebar.checkbox("Apply Reclaim Rate Filter")
    reclaim_rate = st.sidebar.number_input(
        "Minimum Reclaim Rate Threshold (%):", 
        min_value=0.0, 
        max_value=100.0, 
        value=15.0,
        disabled=not apply_reclaim_filter
    )
    
    # Rating filter
    apply_rating_filter = st.sidebar.checkbox("Apply Rating Filter")
    max_rating = st.sidebar.number_input(
        "Maximum Rating:", 
        min_value=0.0, 
        max_value=10.0, 
        value=3.0,
        disabled=not apply_rating_filter
    )

    # Return order rate filter
    apply_return_order_filter = st.sidebar.checkbox("Apply Return Order Rate Filter")
    return_order_rate = st.sidebar.number_input(
        "Minimum Return Order Rate Threshold (%):",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        disabled=not apply_return_order_filter
    )

    # Reclaim order rate filter
    apply_reclaim_order_filter = st.sidebar.checkbox("Apply Reclaim Order Rate Filter")
    reclaim_order_rate = st.sidebar.number_input(
        "Minimum Reclaim Order Rate Threshold (%):",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        disabled=not apply_reclaim_order_filter
    )

    # Available on stock filter
    apply_available_stock_filter = st.sidebar.checkbox("Apply Available on Stock Filter")
    min_available_stock = st.sidebar.number_input(
        "Minimum Available on Stock:",
        min_value=0,
        value=10,
        disabled=not apply_available_stock_filter
    )

    # Value available on stock filter
    apply_value_stock_filter = st.sidebar.checkbox("Apply Value Available on Stock Filter")
    min_value_stock = st.sidebar.number_input(
        "Minimum Value Available on Stock:",
        min_value=0.0,
        value=1000.0,
        disabled=not apply_value_stock_filter
    )

    # Avg return product count filter
    apply_avg_return_count_filter = st.sidebar.checkbox("Apply Avg Return Product Count Filter")
    max_avg_return_count = st.sidebar.number_input(
        "Maximum Avg Return Product Count:",
        min_value=0.0,
        value=2.0,
        disabled=not apply_avg_return_count_filter
    )

    # Avg reclaim product count filter
    apply_avg_reclaim_count_filter = st.sidebar.checkbox("Apply Avg Reclaim Product Count Filter")
    max_avg_reclaim_count = st.sidebar.number_input(
        "Maximum Avg Reclaim Product Count:",
        min_value=0.0,
        value=2.0,
        disabled=not apply_avg_reclaim_count_filter
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
        'apply_return_order_filter': apply_return_order_filter,
        'return_order_rate': return_order_rate,
        'apply_reclaim_order_filter': apply_reclaim_order_filter,
        'reclaim_order_rate': reclaim_order_rate,
        'apply_value_stock_filter': apply_value_stock_filter,
        'min_value_stock': min_value_stock,
        'apply_available_stock_filter': apply_available_stock_filter,
        'min_available_stock': min_available_stock,
        'apply_avg_return_count_filter': apply_avg_return_count_filter,
        'max_avg_return_count': max_avg_return_count,
        'apply_avg_reclaim_count_filter': apply_avg_reclaim_count_filter,
        'max_avg_reclaim_count': max_avg_reclaim_count,
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
            st.dataframe(display_df, width='stretch')
            
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

def show_metrics_page():
    """Display the metrics definitions page"""
    st.header("📚 Metric Definitions")
    st.markdown("This page explains the key metrics used in the Product Quality Analysis Dashboard.")
    
    # Order Metrics
    st.subheader("📦 Order Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**unique_orders_count**")
        st.write("Total number of unique orders containing this product")
        
        st.markdown("**unique_orders_with_return**")
        st.write("Number of unique orders where this product was returned")
        
        st.markdown("**unique_orders_with_reclaim**")
        st.write("Number of unique orders where this product was reclaimed due to damage/defects")
    
    with col2:
        st.markdown("**return_order_rate**")
        st.write("Percentage of orders that resulted in a product return: (unique_orders_with_return / unique_orders_count) × 100")
        
        st.markdown("**reclaim_order_rate**")
        st.write("Percentage of orders that resulted in a product reclaim: (unique_orders_with_reclaim / unique_orders_count) × 100")
    
    # Product Count Metrics
    st.subheader("📊 Product Count Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**amount_returned_sortimodel**")
        st.write("Total quantity of this product returned (with conditions according to sortiment model (everything except 'BONAMI_*'))")
        
        st.markdown("**amount_reclaimed_sortimodel**")
        st.write("Total quantity of this product reclaimed due to defects (with conditions according to sortiment model (everything except 'BONAMI_*'))")
    
    with col2:
        st.markdown("**avg_return_product_count**")
        st.write("Average number of products returned per return order")
        
        st.markdown("**avg_reclaim_product_count**")
        st.write("Average number of products reclaimed per reclaim order")
    
    # Rate Metrics
    st.subheader("📈 Return & Reclaim Rate Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**return_rate_lifetime**")
        st.write("Percentage of total product units that were returned over the product's entire lifetime")
        
        st.markdown("**return_rate_365**")
        st.write("Percentage of product units returned in the last 365 days")
        
        st.markdown("**return_rate_180**")
        st.write("Percentage of product units returned in the last 180 days")
    
    with col2:
        st.markdown("**reclaim_rate_lifetime**")
        st.write("Percentage of total product units that were reclaimed over the product's entire lifetime")
        
        st.markdown("**reclaim_rate_365**")
        st.write("Percentage of product units reclaimed in the last 365 days")
        
        st.markdown("**reclaim_rate_180**")
        st.write("Percentage of product units reclaimed in the last 180 days")
    
    # Product Lifecycle & Rating Metrics
    st.subheader("🕒 Product Lifecycle & Quality Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**days_since_launch**")
        st.write("Number of days since the product was first made available for sale")
        
        st.markdown("**low_rating_percentage**")
        st.write("Percentage of ratings that are considered 'low' (up to 3 stars)")
    
    # Business Impact Metrics
    st.subheader("💰 Business Impact Metrics")
    
    st.markdown("**return_impact**")
    st.write("Financial impact of returns calculated as: amount_lt × gm1a_current × (return_rate_lifetime ÷ 100)")
    
    st.markdown("**reclaim_impact**")
    st.write("Financial impact of reclaims calculated as: amount_lt × gm1a_current × (reclaim_rate_lifetime ÷ 100)")
    
    st.markdown("**total_business_impact**")
    st.write("Combined financial impact: return_impact + reclaim_impact")
    
if __name__ == "__main__":
    main()