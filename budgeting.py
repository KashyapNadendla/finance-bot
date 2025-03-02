import streamlit as st

def budgeting_tool():
    st.header("Budgeting Tool")
    income = st.number_input("Monthly Income", min_value=0.0, step=100.0,
                             help="Your total monthly income after taxes.")
    expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0,
                               help="Your total monthly expenses.")
    savings = income - expenses
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")
