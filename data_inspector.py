#!/usr/bin/env python3
"""
Data Inspector for Substance Overdose Mortality Dataset

This utility helps inspect and understand the structure of the new 2015-2023 dataset.
Use this to debug data loading issues and understand the data format.

Usage:
    python data_inspector.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Data path
DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'

def inspect_excel_file():
    """Inspect the Excel file structure"""
    print("="*80)
    print("EXCEL FILE INSPECTION")
    print("="*80)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Please check the file path and ensure the data file exists.")
        return None
    
    try:
        # Read the Excel file
        df = pd.read_excel(DATA_PATH)
        
        print(f"✅ Successfully loaded: {DATA_PATH}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print()
        
        # Display basic info
        print("COLUMN INFORMATION:")
        print("-" * 40)
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            print(f"{i+1:2d}. {col:<25} | {str(dtype):<10} | {null_count:4d} nulls | {unique_count:4d} unique")
        
        print()
        print("FIRST 10 ROWS:")
        print("-" * 40)
        print(df.head(10).to_string())
        
        print()
        print("LAST 5 ROWS:")
        print("-" * 40)
        print(df.tail(5).to_string())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

def analyze_date_structure(df):
    """Analyze the date structure in the dataset"""
    print("\n" + "="*80)
    print("DATE STRUCTURE ANALYSIS")
    print("="*80)
    
    # Check for date-related columns
    date_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time']):
            date_columns.append(col)
    
    print(f"Date-related columns found: {date_columns}")
    
    for col in date_columns:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Sample values: {df[col].head().tolist()}")
        print(f"  Unique count: {df[col].nunique()}")
        
        if df[col].nunique() < 20:
            print(f"  All unique values: {sorted(df[col].unique())}")
    
    # Try to create a proper date column
    try:
        if 'Row Labels' in df.columns:
            df['Date_Parsed'] = pd.to_datetime(df['Row Labels'])
            print(f"\n✅ Successfully parsed 'Row Labels' as dates")
            print(f"   Date range: {df['Date_Parsed'].min()} to {df['Date_Parsed'].max()}")
            
        elif 'Year_Code' in df.columns and 'Month_Code' in df.columns:
            df['Date_Parsed'] = pd.to_datetime(
                df['Year_Code'].astype(str) + '-' + 
                df['Month_Code'].astype(str).str.zfill(2) + '-01'
            )
            print(f"\n✅ Successfully created dates from Year_Code and Month_Code")
            print(f"   Date range: {df['Date_Parsed'].min()} to {df['Date_Parsed'].max()}")
            
    except Exception as e:
        print(f"\n❌ Error parsing dates: {e}")

def analyze_deaths_data(df):
    """Analyze the deaths/mortality data"""
    print("\n" + "="*80)
    print("DEATHS DATA ANALYSIS")
    print("="*80)
    
    # Find columns that might contain death counts
    death_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['death', 'mortality', 'sum', 'count']):
            death_columns.append(col)
    
    print(f"Death-related columns found: {death_columns}")
    
    for col in death_columns:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Sample values: {df[col].head().tolist()}")
        
        # Check for non-numeric values
        if df[col].dtype == 'object':
            non_numeric = df[col][pd.to_numeric(df[col], errors='coerce').isna()]
            if len(non_numeric) > 0:
                print(f"  Non-numeric values: {non_numeric.unique()}")
        
        # Basic statistics for numeric data
        try:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            print(f"  Statistics:")
            print(f"    Min: {numeric_col.min()}")
            print(f"    Max: {numeric_col.max()}")
            print(f"    Mean: {numeric_col.mean():.1f}")
            print(f"    Median: {numeric_col.median():.1f}")
            print(f"    Null/missing: {numeric_col.isna().sum()}")
            
        except:
            print(f"  Could not compute statistics")

def create_data_visualization(df):
    """Create visualizations to understand the data"""
    print("\n" + "="*80)
    print("DATA VISUALIZATION")
    print("="*80)
    
    try:
        # Create processed data for visualization
        processed_df = df.copy()
        
        # Handle date parsing
        if 'Row Labels' in df.columns:
            processed_df['Date'] = pd.to_datetime(df['Row Labels'])
        elif 'Year_Code' in df.columns and 'Month_Code' in df.columns:
            processed_df['Date'] = pd.to_datetime(
                df['Year_Code'].astype(str) + '-' + 
                df['Month_Code'].astype(str).str.zfill(2) + '-01'
            )
        else:
            print("❌ Could not create date column for visualization")
            return
        
        # Handle deaths data
        if 'Sum of Deaths' in df.columns:
            processed_df['Deaths'] = pd.to_numeric(df['Sum of Deaths'], errors='coerce')
        elif 'Deaths' in df.columns:
            processed_df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
        else:
            print("❌ Could not find deaths column for visualization")
            return
        
        # Clean data
        processed_df = processed_df.dropna(subset=['Date', 'Deaths'])
        processed_df = processed_df.sort_values('Date')
        
        print(f"✅ Processed data for visualization: {len(processed_df)} records")
        print(f"   Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
        print(f"   Deaths range: {processed_df['Deaths'].min()} to {processed_df['Deaths'].max()}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Substance Overdose Mortality Data Analysis (2015-2023)', fontsize=16, fontweight='bold')
        
        # Time series plot
        ax1 = axes[0, 0]
        ax1.plot(processed_df['Date'], processed_df['Deaths'], linewidth=2, color='darkred')
        ax1.set_title('Deaths Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Deaths')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Distribution histogram
        ax2 = axes[0, 1]
        ax2.hist(processed_df['Deaths'], bins=30, alpha=0.7, color='darkblue', edgecolor='black')
        ax2.set_title('Distribution of Monthly Deaths')
        ax2.set_xlabel('Number of Deaths')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Seasonal pattern (if enough data)
        if len(processed_df) >= 12:
            ax3 = axes[1, 0]
            processed_df['Month'] = processed_df['Date'].dt.month
            monthly_avg = processed_df.groupby('Month')['Deaths'].mean()
            
            ax3.bar(monthly_avg.index, monthly_avg.values, color='darkgreen', alpha=0.7)
            ax3.set_title('Average Deaths by Month')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Average Deaths')
            ax3.set_xticks(range(1, 13))
            ax3.grid(True, alpha=0.3)
        
        # Year-over-year comparison
        if len(processed_df) >= 24:
            ax4 = axes[1, 1]
            processed_df['Year'] = processed_df['Date'].dt.year
            yearly_total = processed_df.groupby('Year')['Deaths'].sum()
            
            ax4.bar(yearly_total.index, yearly_total.values, color='darkgoldenrod', alpha=0.7)
            ax4.set_title('Total Deaths by Year')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Total Deaths')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'data_inspection_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        # Show basic trend analysis
        print(f"\nTREND ANALYSIS:")
        print(f"  First year average: {processed_df[processed_df['Year'] == processed_df['Year'].min()]['Deaths'].mean():.0f} deaths/month")
        print(f"  Last year average: {processed_df[processed_df['Year'] == processed_df['Year'].max()]['Deaths'].mean():.0f} deaths/month")
        
        # Calculate year-over-year growth
        if len(yearly_total) > 1:
            total_growth = (yearly_total.iloc[-1] - yearly_total.iloc[0]) / yearly_total.iloc[0] * 100
            print(f"  Total growth ({processed_df['Year'].min()}-{processed_df['Year'].max()}): {total_growth:.1f}%")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def suggest_preprocessing_steps(df):
    """Suggest preprocessing steps based on data inspection"""
    print("\n" + "="*80)
    print("PREPROCESSING RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check date handling
    if 'Row Labels' in df.columns:
        recommendations.append("✅ Use 'Row Labels' column for dates with pd.to_datetime()")
    elif 'Year_Code' in df.columns and 'Month_Code' in df.columns:
        recommendations.append("✅ Combine 'Year_Code' and 'Month_Code' to create date column")
    else:
        recommendations.append("❌ Date handling needs custom solution")
    
    # Check deaths data
    if 'Sum of Deaths' in df.columns:
        recommendations.append("✅ Use 'Sum of Deaths' as primary deaths column")
        
        # Check for suppressed values
        if df['Sum of Deaths'].dtype == 'object':
            non_numeric = df['Sum of Deaths'][pd.to_numeric(df['Sum of Deaths'], errors='coerce').isna()]
            if len(non_numeric) > 0:
                recommendations.append(f"⚠️  Handle non-numeric values in deaths data: {non_numeric.unique()}")
    elif 'Deaths' in df.columns:
        recommendations.append("✅ Use 'Deaths' as primary deaths column")
    else:
        recommendations.append("❌ Deaths column identification needs custom solution")
    
    # Data quality checks
    total_rows = len(df)
    
    for col in df.columns:
        null_pct = df[col].isnull().sum() / total_rows * 100
        if null_pct > 5:
            recommendations.append(f"⚠️  Column '{col}' has {null_pct:.1f}% missing values")
    
    # Suggest data validation
    recommendations.append("✅ Sort data by date and check for gaps")
    recommendations.append("✅ Validate date range covers 2015-2023")
    recommendations.append("✅ Check for duplicate dates")
    
    print("Preprocessing steps to implement:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")

def generate_data_report(df):
    """Generate a comprehensive data report"""
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    if df is None:
        print("❌ Cannot generate report - data loading failed")
        return
    
    report = []
    report.append("SUBSTANCE OVERDOSE MORTALITY DATA REPORT")
    report.append("="*50)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data file: {DATA_PATH}")
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append(f"  Total rows: {len(df):,}")
    report.append(f"  Total columns: {len(df.columns)}")
    report.append(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    report.append("")
    
    # Column summary
    report.append("COLUMNS:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        report.append(f"  {col:<25} | {dtype:<10} | {null_count:4d} nulls ({null_pct:4.1f}%)")
    
    report.append("")
    
    # Data quality issues
    issues = []
    
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct > 10:
            issues.append(f"High missing data in '{col}': {null_pct:.1f}%")
    
    if issues:
        report.append("DATA QUALITY ISSUES:")
        for issue in issues:
            report.append(f"  ⚠️  {issue}")
    else:
        report.append("DATA QUALITY: ✅ No major issues detected")
    
    # Save report
    report_text = "\n".join(report)
    
    with open('data_inspection_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Report saved to: data_inspection_report.txt")

def main():
    """Main inspection function"""
    print("SUBSTANCE OVERDOSE MORTALITY DATA INSPECTOR")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print()
    
    # Load and inspect data
    df = inspect_excel_file()
    
    if df is not None:
        # Detailed analysis
        analyze_date_structure(df)
        analyze_deaths_data(df)
        create_data_visualization(df)
        suggest_preprocessing_steps(df)
        generate_data_report(df)
        
        print("\n" + "="*80)
        print("INSPECTION COMPLETE")
        print("="*80)
        print("Files generated:")
        print("  📊 data_inspection_plots.png")
        print("  📄 data_inspection_report.txt")
        print()
        print("Next steps:")
        print("  1. Review the visualizations to understand data patterns")
        print("  2. Check the preprocessing recommendations")
        print("  3. Run the comprehensive evaluation pipeline")
        
    else:
        print("\n❌ Data inspection failed. Please check the data file and try again.")

if __name__ == "__main__":
    main()