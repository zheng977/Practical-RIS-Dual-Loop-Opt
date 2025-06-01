import pandas as pd
import numpy as np

# Create dataframes from the provided results
ga_data = pd.DataFrame({
    '0.5': [85.620206, 75.824724, 56.023734, 67.113905],
    '1.5': [90.805032, 83.524093, 80.276151, 66.916661],
    '3.0': [91.150492, 87.555212, 77.624737, 59.783703]
}, index=[0.2, 0.4, 0.6, 0.8])
ga_data.index.name = 'alpha_min'
ga_data.columns.name = 'gamma'

sca_data = pd.DataFrame({
    '0.5': [68.815504, 57.156765, 39.415814, 15.718941],
    '1.5': [78.619743, 71.206438, 56.018366, 27.515577],
    '3.0': [83.390299, 75.756225, 62.820088, 33.989961]
}, index=[0.2, 0.4, 0.6, 0.8])
sca_data.index.name = 'alpha_min'
sca_data.columns.name = 'gamma'

no_opt_data = pd.DataFrame({
    '0.5': [-39.400061, -81.043376, -145.342321, -232.114401],
    '1.5': [5.999275, -20.239526, -74.656934, -181.775098],
    '3.0': [27.450169, 0.693247, -46.798869, -155.720771]
}, index=[0.2, 0.4, 0.6, 0.8])
no_opt_data.index.name = 'alpha_min'
no_opt_data.columns.name = 'gamma'

# Method 1: Format for Origin 3D bar chart - Long format
# This format is often preferred for 3D visualization in Origin

def create_long_format(df, algorithm):
    result = []
    for alpha in df.index:
        for gamma in df.columns:
            result.append({
                'Algorithm': algorithm,
                'Alpha_Min': alpha,
                'Gamma': float(gamma),
                'Improvement': df.loc[alpha, gamma]
            })
    return pd.DataFrame(result)

# Combine all algorithms
long_format = pd.concat([
    create_long_format(ga_data, 'GA'),
    create_long_format(sca_data, 'SCA'),
    create_long_format(no_opt_data, 'No_Optimization')
])

# Save long format for Origin
long_format.to_csv('improvement_data_long_format.csv', index=False)

# Method 2: Format for Origin with separate sheets per algorithm
with pd.ExcelWriter('improvement_data_by_algorithm.xlsx') as writer:
    ga_data.to_excel(writer, sheet_name='GA')
    sca_data.to_excel(writer, sheet_name='SCA')
    no_opt_data.to_excel(writer, sheet_name='No_Optimization')

# Method 3: Prepare stacked data for grouped bar charts
# Reshape data for comparison between algorithms
stacked_data = pd.DataFrame()

for alpha in ga_data.index:
    for gamma in ga_data.columns:
        row_name = f"Alpha_{alpha}_Gamma_{gamma}"
        stacked_data.loc[row_name, 'Alpha_Min'] = alpha
        stacked_data.loc[row_name, 'Gamma'] = float(gamma)
        stacked_data.loc[row_name, 'GA'] = ga_data.loc[alpha, gamma]
        stacked_data.loc[row_name, 'SCA'] = sca_data.loc[alpha, gamma]
        stacked_data.loc[row_name, 'No_Optimization'] = no_opt_data.loc[alpha, gamma]

# Save stacked format
stacked_data.to_csv('improvement_data_stacked.csv', index=False)

# Method 4: Format for heatmap comparison
# Prepare difference between GA and SCA
diff_data = ga_data - sca_data
diff_data.to_csv('ga_vs_sca_difference.csv')

print("Files created:")
print("1. improvement_data_long_format.csv - Best for Origin 3D bar charts")
print("2. improvement_data_by_algorithm.xlsx - Separate sheets per algorithm")
print("3. improvement_data_stacked.csv - Good for grouped bar comparisons")
print("4. ga_vs_sca_difference.csv - Shows GA improvement over SCA")

# Display sample of the long format data (most useful for Origin)
print("\nSample of long format data (first 10 rows):")
print(long_format.head(10))