# import streamlit as st

# def budgeting_tool():
#     st.header("Budgeting Tool")
#     income = st.number_input("Monthly Income", min_value=0.0, step=100.0,
#                              help="Your total monthly income after taxes.")
#     expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0,
#                                help="Your total monthly expenses.")
#     savings = income - expenses
#     if savings >= 0:
#         st.success(f"Monthly Savings: ${savings:.2f}")
#     else:
#         st.error(f"Monthly Deficit: ${-savings:.2f}")

import streamlit as st
import matplotlib.pyplot as plt

def budgeting_tool():
    st.header("Budgeting Tool")
    
    # Input for monthly income
    income = st.number_input("Monthly Income", min_value=0.0, step=100.0,
                             help="Your total monthly income after taxes.")
    
    st.markdown("### Expense Categories")
    # Define a set of common expense categories
    expense_categories = ["Rent", "Utilities", "Food", "Transportation", "Entertainment", "Others"]
    
    expenses = {}
    total_expenses = 0.0
    # Create number inputs for each expense category
    for cat in expense_categories:
        expense = st.number_input(f"{cat} Expense", min_value=0.0, step=10.0, key=cat)
        expenses[cat] = expense
        total_expenses += expense
        
    st.write(f"**Total Expenses:** ${total_expenses:.2f}")
    
    savings = income - total_expenses
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")
    
    # Optional: Display a pie chart of the expense breakdown
    if total_expenses > 0:
        fig, ax = plt.subplots()
        ax.pie(expenses.values(), labels=expenses.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
