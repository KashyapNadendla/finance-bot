import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def budgeting_tool():
    st.header("Enhanced Budgeting Tool")

    # Monthly Income Input
    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        step=100.0,
        help="Your total monthly income after taxes."
    )

    # Expense Categories
    st.subheader("Enter Your Monthly Expenses by Category")
    expense_categories = {
        'Housing (Rent/Mortgage)': 0.0,
        'Utilities': 0.0,
        'Food': 0.0,
        'Transportation': 0.0,
        'Entertainment': 0.0,
        'Healthcare': 0.0,
        'Insurance': 0.0,
        'Debt Payments': 0.0,
        'Education': 0.0,
        'Savings & Investments': 0.0,
        'Miscellaneous': 0.0,
    }

    total_expenses = 0.0

    # Input expenses for each category
    for category in expense_categories:
        expense = st.number_input(
            f"{category}",
            min_value=0.0,
            step=10.0,
            help=f"Enter monthly expenses for {category.lower()}."
        )
        expense_categories[category] = expense
        total_expenses += expense

    # Calculate Savings
    savings = income - total_expenses

    # Display Results
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")

    st.subheader("Expenses Breakdown")
    if income > 0:
        expense_percentages = {cat: (amt / income) * 100 for cat, amt in expense_categories.items()}
        df_expenses = pd.DataFrame({
            'Category': list(expense_categories.keys()),
            'Amount': list(expense_categories.values()),
            'Percentage of Income': [f"{p:.2f}%" for p in expense_percentages.values()]
        })
        st.dataframe(df_expenses)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax1.pie(
            expense_categories.values(),
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            pctdistance=0.85
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig1.gca().add_artist(centre_circle)
        ax1.legend(
            wedges,
            expense_categories.keys(),
            title="Expense Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10
        )
        ax1.axis('equal')
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.warning("Please enter your income to see expense breakdown.")

    # Savings Goals
    st.subheader("Savings Goals")
    goal_amount = st.number_input(
        "Enter your savings goal amount:",
        min_value=0.0,
        step=100.0,
        help="Enter the total amount you aim to save."
    )
    goal_timeframe = st.number_input(
        "Enter your goal timeframe in months:",
        min_value=1,
        step=1,
        help="Enter the number of months you plan to achieve this goal."
    )

    if savings > 0 and goal_amount > 0 and goal_timeframe > 0:
        monthly_savings_needed = goal_amount / goal_timeframe
        if savings >= monthly_savings_needed:
            st.success(f"You are on track! You need ${monthly_savings_needed:.2f} per month to hit your goal.")
        else:
            st.warning(f"You need ${monthly_savings_needed:.2f} per month to meet your goal. Consider adjustments.")
    elif goal_amount > 0 and goal_timeframe > 0:
        st.error("Your current budget does not allow for savings towards your goal.")

    # Budget Recommendations based on 50/30/20 rule
    st.subheader("Budget Recommendations")
    if income > 0:
        needs = income * 0.5
        wants = income * 0.3
        savings_recommended = income * 0.2

        st.write("Based on the 50/30/20 rule:")
        st.write(f"- **Needs (50%)**: ${needs:.2f}")
        st.write(f"- **Wants (30%)**: ${wants:.2f}")
        st.write(f"- **Savings (20%)**: ${savings_recommended:.2f}")

        total_needs = sum([expense_categories[cat] for cat in ['Housing (Rent/Mortgage)', 'Utilities', 'Food', 'Transportation', 'Healthcare', 'Insurance', 'Debt Payments']])
        total_wants = sum([expense_categories[cat] for cat in ['Entertainment', 'Education', 'Miscellaneous']])
        total_savings = savings if savings > 0 else 0

        st.write("Your actual spending:")
        st.write(f"- **Needs**: ${total_needs:.2f}")
        st.write(f"- **Wants**: ${total_wants:.2f}")
        st.write(f"- **Savings**: ${total_savings:.2f}")

        labels = ['Needs', 'Wants', 'Savings']
        recommended = [needs, wants, savings_recommended]
        actual = [total_needs, total_wants, total_savings]

        x = np.arange(len(labels))
        width = 0.35

        fig2, ax2 = plt.subplots()
        rects1 = ax2.bar(x - width/2, recommended, width, label='Recommended')
        rects2 = ax2.bar(x + width/2, actual, width, label='Actual')

        ax2.set_ylabel('Amount ($)')
        ax2.set_title('Recommended vs. Actual Spending')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(
                    f'${height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom'
                )

        autolabel(rects1)
        autolabel(rects2)

        st.pyplot(fig2)
    else:
        st.warning("Please enter your income to see budget recommendations.")

    # Future Projections
    st.subheader("Future Savings Projection")
    projection_months = st.number_input(
        "Enter the number of months for projection:",
        min_value=1,
        step=1,
        help="Enter how many months into the future you'd like to project your savings."
    )
    if savings > 0 and projection_months > 0:
        projected_savings = savings * projection_months
        st.write(f"In {projection_months} months, you could save about ${projected_savings:.2f} if your situation remains unchanged.")
    elif projection_months > 0:
        st.error("Your current budget does not allow for savings projection.")
