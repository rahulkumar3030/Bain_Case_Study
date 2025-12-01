"""
analysis_utils.py
Core analysis functions for employee attrition visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Grouping configuration
GROUPING_CONFIG = {
    'PerformanceRating': {
        1: 'Bad',
        2: 'Bad',
        3: 'Average',
        4: 'Great',
        5: 'Great'
    },
    'JobSatisfaction': {
        1: 'Bad',
        2: 'Bad',
        3: 'Average',
        4: 'Great',
        5: 'Great'
    },
    'WorkLifeBalance': {
        1: 'Bad',
        2: 'Bad',
        3: 'Average',
        4: 'Great',
        5: 'Great'
    },
    'EnvironmentSatisfaction': {
        1: 'Bad',
        2: 'Bad',
        3: 'Average',
        4: 'Great',
        5: 'Great'
    },
    'YearsAtCompany': {
        'logic': lambda x: 'New (0-2 yrs)' if x <= 2 
                          else 'Mid (3-5 yrs)' if x <= 5 
                          else 'Experienced (6-10 yrs)' if x <= 10 
                          else 'Veteran (10+ yrs)'
    },
    'YearsInCurrentRole': {
        'logic': lambda x: '0-2 yrs' if x <= 2 
                          else '3-5 yrs' if x <= 5 
                          else '6-10 yrs' if x <= 10 
                          else '10+ yrs'
    },
    'Age': {
        'logic': lambda x: 'Young (18-30)' if x <= 30 
                          else 'Mid-Career (31-40)' if x <= 40 
                          else 'Senior (41-50)' if x <= 50 
                          else 'Experienced (50+)'
    },
    'MonthlyIncome': {
        'logic': lambda x: 'Low (<5K)' if x < 5000 
                          else 'Medium (5K-10K)' if x < 10000 
                          else 'High (10K-15K)' if x < 15000 
                          else 'Very High (15K+)'
    }
}

# Column display names
COLUMN_DISPLAY_NAMES = {
    'YearsAtCompany': 'Tenure (Years at Company)',
    'YearsInCurrentRole': 'Years in Current Role',
    'Age': 'Age Group',
    'MonthlyIncome': 'Monthly Income',
    'DistanceFromHome': 'Distance from Home',
    'PerformanceRating': 'Performance Rating',
    'JobSatisfaction': 'Job Satisfaction',
    'WorkLifeBalance': 'Work-Life Balance',
    'EnvironmentSatisfaction': 'Environment Satisfaction',
    'OverTime': 'Overtime Status',
    'Department': 'Department',
    'JobRole': 'Job Role',
    'Gender': 'Gender',
    'EducationLevel': 'Education Level',
    'TrainingTimesLastYear': 'Training Frequency'
}

# Insights text for each column
COLUMN_INSIGHTS = {
    'JobSatisfaction': """
    **Key Insight:** Low job satisfaction is the strongest predictor of attrition. 
    Employees in the "Bad" category show dramatically higher attrition rates compared to 
    those with "Average" or "Great" satisfaction.
    
    **Recommendation:** Conduct immediate engagement surveys and one-on-ones with low 
    satisfaction employees to identify root causes.
    """,
    
    'OverTime': """
    **Key Insight:** Employees working overtime have significantly higher attrition rates. 
    This indicates workload imbalance and potential burnout.
    
    **Recommendation:** Audit overtime distribution, implement mandatory rest periods, 
    and consider additional hiring in high-overtime departments.
    """,
    
    'YearsAtCompany': """
    **Key Insight:** New employees (0-2 years) face the highest attrition risk, 
    indicating potential issues with onboarding, role clarity, or cultural fit.
    
    **Recommendation:** Strengthen onboarding programs, assign mentors to new hires, 
    and conduct regular stay interviews at 3, 6, and 12-month marks.
    """,
    
    'WorkLifeBalance': """
    **Key Insight:** Poor work-life balance nearly doubles attrition risk. 
    This is a critical retention factor that compounds with overtime.
    
    **Recommendation:** Introduce flexible work policies, review workload distribution, 
    and promote better time management practices.
    """,
    
    'PerformanceRating': """
    **Key Insight:** Lower-rated employees show higher attrition, but this may indicate 
    performance management gaps rather than employee quality issues.
    
    **Recommendation:** Review performance feedback processes, ensure constructive 
    coaching, and provide clear improvement pathways.
    """,
    
    'Department': """
    **Key Insight:** Some departments show higher attrition than others, pointing to 
    local factors like leadership, workload, or growth opportunities.
    
    **Recommendation:** Investigate high-attrition departments for management issues, 
    resource constraints, or career development gaps.
    """,
    
    'JobRole': """
    **Key Insight:** Certain roles have higher attrition, possibly due to market 
    competitiveness, role design, or limited career progression.
    
    **Recommendation:** Review compensation competitiveness, role expectations, 
    and career paths for high-attrition roles.
    """,
    
    'Age': """
    **Key Insight:** Attrition patterns vary by age group, with different factors 
    driving turnover at different career stages.
    
    **Recommendation:** Tailor retention strategies by age group (e.g., growth 
    opportunities for younger, work-life balance for older).
    """,
    
    'Gender': """
    **Key Insight:** Gender-based attrition differences may indicate equity or 
    inclusion issues that need attention.
    
    **Recommendation:** Review policies for pay equity, promotion fairness, 
    and inclusive workplace practices.
    """,
    
    'MonthlyIncome': """
    **Key Insight:** Income levels correlate with attrition, but the relationship 
    may not be linear—market competitiveness matters.
    
    **Recommendation:** Benchmark salaries against market rates and ensure 
    internal pay equity.
    """,
    
    'EnvironmentSatisfaction': """
    **Key Insight:** Poor environmental satisfaction (workplace conditions, tools, 
    facilities) drives attrition.
    
    **Recommendation:** Invest in workplace improvements, modern tools, 
    and comfortable work environments.
    """,
    
    'YearsInCurrentRole': """
    **Key Insight:** Time in current role affects attrition—too long may indicate 
    stagnation, too short may indicate poor fit.
    
    **Recommendation:** Ensure regular role rotations, promotions, and skill 
    development opportunities.
    """,
    
    'EducationLevel': """
    **Key Insight:** Education level may correlate with attrition due to 
    overqualification or career expectations.
    
    **Recommendation:** Align roles with educational backgrounds and provide 
    appropriate growth challenges.
    """,
    
    'TrainingTimesLastYear': """
    **Key Insight:** Training frequency impacts retention—both too little and 
    too much may be problematic.
    
    **Recommendation:** Ensure relevant, meaningful training aligned with 
    career development goals.
    """,
    
    'DistanceFromHome': """
    **Key Insight:** Long commutes increase attrition risk due to time, cost, 
    and work-life balance impacts.
    
    **Recommendation:** Offer remote work options or flexible schedules for 
    employees with long commutes.
    """
}


def analyze_attrition(df, column_name, top_n=None):
    """
    Analyze attrition by any column with automatic grouping
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found")
    
    df_work = df.copy()
    
    # Apply grouping if configured
    if column_name in GROUPING_CONFIG:
        config = GROUPING_CONFIG[column_name]
        analysis_col = f"{column_name}_Grouped"
        
        if 'logic' in config:
            df_work[analysis_col] = df_work[column_name].apply(config['logic'])
        else:
            df_work[analysis_col] = df_work[column_name].map(config)
    else:
        analysis_col = column_name
    
    # Calculate metrics
    summary = (
        df_work.groupby(analysis_col, dropna=False)
        .agg(
            Total=('EmployeeID', 'count'),
            Attrited=('Attrition', lambda x: (x == 'Yes').sum())
        )
        .reset_index()
    )
    
    summary['Retained'] = summary['Total'] - summary['Attrited']
    summary['Attrition_Rate'] = (summary['Attrited'] / summary['Total'] * 100).round(2)
    summary = summary.sort_values('Attrition_Rate', ascending=False)
    
    if top_n:
        summary = summary.head(top_n)
    
    summary.rename(columns={analysis_col: 'Category'}, inplace=True)
    
    return summary


def create_attrition_plot(df, column_name, top_n=None):
    """
    Create matplotlib figure for attrition visualization
    Returns: (fig, summary_df)
    """
    summary = analyze_attrition(df, column_name, top_n)
    
    n_categories = len(summary)
    use_horizontal = n_categories > 7
    
    # Determine figure size
    if use_horizontal:
        figsize = (10, max(6, n_categories * 0.5))
    else:
        figsize = (max(10, n_categories * 1.2), 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color gradient
    min_rate = summary['Attrition_Rate'].min()
    max_rate = summary['Attrition_Rate'].max()
    
    if max_rate > min_rate:
        norm_rates = (summary['Attrition_Rate'] - min_rate) / (max_rate - min_rate)
    else:
        norm_rates = np.ones(len(summary)) * 0.5
    
    colors = plt.cm.RdYlGn_r(norm_rates)
    
    # Plot
    if use_horizontal:
        bars = ax.barh(range(n_categories), summary['Attrition_Rate'], color=colors)
        ax.set_yticks(range(n_categories))
        ax.set_yticklabels(summary['Category'])
        ax.set_xlabel('Attrition Rate (%)', fontsize=11)
        ax.invert_yaxis()
        
        for i, (bar, rate, total) in enumerate(zip(bars, summary['Attrition_Rate'], summary['Total'])):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{rate:.1f}% (n={total})',
                    va='center', fontsize=9)
    else:
        bars = ax.bar(range(n_categories), summary['Attrition_Rate'], color=colors)
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(summary['Category'], rotation=45, ha='right')
        ax.set_ylabel('Attrition Rate (%)', fontsize=11)
        
        for i, (bar, rate, total) in enumerate(zip(bars, summary['Attrition_Rate'], summary['Total'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%\n(n={total})',
                    ha='center', va='bottom', fontsize=9)
    
    # Title
    display_name = COLUMN_DISPLAY_NAMES.get(column_name, column_name)
    title = f'Attrition Rate by {display_name}'
    if top_n:
        title += f' (Top {top_n})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Color bar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn_r,
        norm=plt.Normalize(vmin=min_rate, vmax=max_rate)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Attrition Rate (%)', rotation=270, labelpad=20, fontsize=9)
    
    plt.tight_layout()
    
    return fig, summary


def get_insight_text(column_name):
    """Get predefined insight text for a column"""
    return COLUMN_INSIGHTS.get(column_name, "No specific insight available for this column.")
