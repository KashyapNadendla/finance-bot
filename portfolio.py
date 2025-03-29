import streamlit as st
import pandas as pd

def portfolio_manager():
    st.header("Portfolio Manager")

    # Initialize portfolio in session state if it doesn't exist.
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # Form to add a new asset.
    with st.form("add_asset"):
        st.subheader("Add a New Asset")
        asset_symbol = st.text_input("Asset Symbol (e.g. AAPL, BTC)")
        asset_type = st.selectbox("Asset Type", options=["Stock", "Crypto"])
        quantity = st.number_input("Quantity", min_value=0.0, step=0.01)
        purchase_price = st.number_input("Purchase Price", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Add to Portfolio")

        if submitted:
            if asset_symbol:
                new_asset = {
                    "Asset": asset_symbol.upper(),
                    "Type": asset_type,
                    "Quantity": quantity,
                    "Purchase Price": purchase_price
                }
                st.session_state["portfolio"].append(new_asset)
                st.success(f"Added {asset_symbol.upper()} to your portfolio.")
            else:
                st.error("Please enter a valid asset symbol.")

    # Display the current portfolio.
    if st.session_state["portfolio"]:
        df_portfolio = pd.DataFrame(st.session_state["portfolio"])
        st.subheader("Your Portfolio")
        st.dataframe(df_portfolio)
    else:
        st.write("Your portfolio is currently empty.")
