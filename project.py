import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')


file_path = r"C:\Users\DELL\OneDrive\Desktop\Crime_Incidents_in_2024.csv"
df = pd.read_csv(file_path)
print(df)


print("Available columns:")
print(df.columns.tolist())

date_columns = [col for col in df.columns if 'date' in col.lower()]
print("\nDetected date-related columns:", date_columns)


for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Drop rows with missing dates (optional – change depending on important fields)
df.dropna(subset=date_columns, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop unwanted columns if present
columns_to_drop = ['objectid', 'octo_record_id']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Final info
print("\nCleaned Data Info:")
print(df.info())


print("All column names in your dataset:")
print(df.columns.tolist())

#
top_crimes = df['OFFENSE'].value_counts().head(10)

print("Top 10 Most Common Crimes:")
print(top_crimes)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis")
plt.title("Top 10 Most Common Crimes in 2024")
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")
plt.tight_layout()
# plt.show()


# -------------------- Part 2: Most Common Crime Types --------------------

# Count the top 10 most frequent crimes
top_crimes = df['OFFENSE'].value_counts().head(10)

# Print the top crimes
print("\nTop 10 Most Common Crimes in 2024:")
print(top_crimes)

# Plot the top crimes
plt.figure(figsize=(12, 6))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="Reds_r")
plt.title("Top 10 Most Common Crimes in 2024")
plt.xlabel("Number of Reports")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

# -------------------- Part 3: Monthly Crime Trends --------------------

# Ensure correct datetime format
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], errors='coerce')
df = df.dropna(subset=['REPORT_DAT'])

# Extract month from the report date
df['Month'] = df['REPORT_DAT'].dt.month_name()

# Count crimes per month
monthly_crimes = df['Month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

# # Print crime counts by month
print("\nCrime counts by month:")
print(monthly_crimes)

# Plot monthly crime trends
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_crimes.index, y=monthly_crimes.values, palette="Blues_d")
plt.title("Monthly Crime Rates in 2024")
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Top 10 Locations with Most Crime Incidents 

# Get top 10 blocks with highest number of crime reports
top_blocks = df['BLOCK'].value_counts().head(10)

# Print the result
print("\nTop 10 Locations (Blocks) with Most Crime Incidents:")
print(top_blocks)

# Horizontal bar chart using seaborn
plt.figure(figsize=(12, 6))

sns.barplot(y=top_blocks.index, x=top_blocks.values, palette="Greens_r")

plt.title("Top 10 Locations with Most Crime Incidents (2024)")

plt.xlabel("Number of Reports")

plt.ylabel("Location (Block)")

plt.tight_layout()

plt.show()




# -------------------- Part 5: Crime Locations on a Map --------------------

# Drop rows with missing coordinates
df_map = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Plotting the scatter plot of crime locations
plt.figure(figsize=(10, 8))
plt.scatter(
    df_map['LONGITUDE'], 
    df_map['LATITUDE'], 
    alpha=0.3, 
    s=10, 
    c='red'
)
plt.title("Crime Locations in Washington DC (2024)")

plt.xlabel("Longitude")

plt.ylabel("Latitude")



plt.grid(True)

plt.tight_layout()

plt.show()

# -------------------- Part 6: Monthly Crime Trend --------------------

# Ensure REPORT_DAT is in datetime format
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], errors='coerce')

# Extract month and count incidents
df['Month'] = df['REPORT_DAT'].dt.month_name()
monthly_trends = df['Month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

# Drop NaN values (in case some months had no entries)
monthly_trends = monthly_trends.dropna()

# Plot line chart
plt.figure(figsize=(12, 6))
plt.plot(monthly_trends.index, monthly_trends.values, marker='o', linestyle='-', color='purple')
plt.title("Monthly Crime Trend in 2024")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Calculate correlation matrix (only numerical columns)
correlation_matrix = df.corr(numeric_only=True)

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Plotting the heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='white')

plt.title("Correlation Heatmap of Numerical Features", fontsize=16)

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()




