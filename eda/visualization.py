import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

color = sns.color_palette("tab10")

def plot_features_distribution(df: pd.DataFrame):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    axs = axs.flatten()
    
    for i in range(1, len(df.columns)):
        sns.boxplot(x=df.columns[i], data=df, ax=axs[i-1], orient='v')
    
    plt.suptitle('Features Distribution', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def bike_rental_hour(hour_df: pd.DataFrame):
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='hr', y='cnt', data=hour_df)
    sns.set_style('darkgrid')
    plt.title('Box Plot of Bike Rentals by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Count of Bike Rentals')
    plt.show()


def bike_rental_day(hour_df: pd.DataFrame):
    fig, axs = plt.subplots(1, 1, figsize=(14,6))
    sns.boxplot(x='weekday', y='cnt', data=hour_df)
    sns.set_style('darkgrid')
    weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    axs.set_xlabel('Name of the day')
    axs.set_xticklabels(weekday)
    axs.set_ylabel('Count of bike rentals')
    plt.title('Box Plot of Bike Rentals by the day of the week')
    plt.show()


def bike_rental_month(hour_df: pd.DataFrame): 
    fig, axs = plt.subplots(1, 1, figsize=(14,6))
    sns.boxplot(x='mnth', y='cnt', data=hour_df)
    sns.set_style('darkgrid')
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axs.set_xlabel('Month')
    axs.set_xticklabels(month_list)
    axs.set_ylabel('Count of bike rentals')
    plt.title('Box Plot of Bike Rentals by month')
    plt.show()  
    

def average_several_conditions(hour_df: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    color = sns.color_palette("tab10")
    sns.boxplot(data=hour_df, x='holiday', y='cnt', ax=axs[0], palette=color)
    sns.boxplot(data=hour_df, x='weathersit', y='cnt', ax=axs[1], palette=color)
    sns.boxplot(data=hour_df, x='season', y='cnt', ax=axs[2], palette=color)
    plt.suptitle('Impact of Several Conditions on Bike Sharing Count Average', fontsize=16)
    plt.tight_layout()
    plt.show()


def numerical_features(hour_df: pd.DataFrame):
    color = sns.color_palette("tab10")
    temp_min, temp_max = -8, 39
    atemp_min, atemp_max = -16, 50
    original_atemp = hour_df['atemp']*(atemp_max-atemp_min)+atemp_min
    original_temp = hour_df['temp']*(temp_max-temp_min)+temp_min
    original_hum = hour_df['hum']*100
    original_windspeed = hour_df['windspeed']*67

    fig, axs = plt.subplots(2, 2, figsize=(15,8))
    axs.flatten()

    sns.scatterplot(data=hour_df, x=original_atemp, y='cnt', ax=axs[0,0])
    sns.regplot(data=hour_df, x=original_atemp, y='cnt', ax=axs[0, 0], scatter=False, color=color[3])

    sns.scatterplot(data=hour_df, x=original_temp, y='cnt', ax=axs[0,1])
    sns.regplot(data=hour_df, x=original_temp, y='cnt', ax=axs[0, 1], scatter=False, color=color[3])

    sns.scatterplot(data=hour_df, x=original_hum, y='cnt', ax=axs[1,0])
    sns.regplot(data=hour_df, x=original_hum, y='cnt', ax=axs[1, 0], scatter=False, color=color[3])

    sns.scatterplot(data=hour_df, x=original_windspeed, y='cnt', ax=axs[1,1])
    sns.regplot(data=hour_df, x=original_windspeed, y='cnt', ax=axs[1, 1], scatter=False, color=color[3])

    plt.suptitle('Scatter Plots of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.show()


def count_weekdays_weekends(hour_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.pointplot(data=hour_df, x='hr', y='cnt', hue='weekday', palette=color, ax=ax)
    plt.title('Count of bikes during weekdays and weekends', fontsize=16)
    plt.show()


def count_each_year(hour_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.pointplot(data=hour_df, x='mnth', y='cnt', hue='year', palette=color, ax=ax)
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_list)
    plt.title('Count of bikes each year', fontsize=16)
    plt.show()


def seasonwise_distribution(hour_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.pointplot(data=hour_df, x='hr', y='cnt', hue='season', palette=color, ax=ax)
    plt.title('Season wise hourly distribution of counts', fontsize=16)
    plt.show()


def impact_of_holidays(hour_df: pd.DataFrame):
    y = hour_df.groupby('holiday')['cnt'].mean().reset_index()
    c = hour_df[hour_df.holiday==1].groupby('hr')['cnt'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(x ='holiday', y ='cnt', data = y).set_title('Average Bike demand during Holidays')
    is_holiday = ['No', 'Yes']

    ax.set_xlabel('Holiday')
    ax.set_xticklabels(is_holiday)
    ax.set_ylabel('Count of bike rentals')
    plt.show()

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x ='hr', y ='cnt', data = c).set_title('Hourwise Bike sharing Demand during holidays')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Count of bike rentals')
    plt.show()


def spearman_correlation(hour_df: pd.DataFrame):
    plt.figure(figsize=(14,7))
    sns.heatmap(hour_df.corr('spearman'), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
    plt.title('Spearman Correlation Heatmap', fontsize=16)
    plt.show()


def outlier_detection(hour_df: pd.DataFrame):
    # Outlier detection using a boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(hour_df['cnt'])
    plt.title('Boxplot of Bike Rental Counts')
    plt.ylabel('Count')
    plt.show()

    # Calculate z-scores for the 'cnt' column
    mean_cnt = hour_df['cnt'].mean()
    std_cnt = hour_df['cnt'].std()
    z_scores = (hour_df['cnt'] - mean_cnt) / std_cnt

    # Set a threshold for outlier detection (e.g., z-score > 3)
    outlier_threshold = 3
    outliers = hour_df[z_scores > outlier_threshold]

    # Visualize the data and highlight outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(hour_df.index, hour_df['cnt'], label='Bike Rentals')
    plt.scatter(outliers.index, outliers['cnt'], color='red', label='Outliers')
    plt.title('Bike Rental Counts with Outliers Highlighted')
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def identify_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def summarize_outliers(data):
    summary = []
    for column in data.columns:
        if column in ['dteday', 'holiday']:
            continue
        outliers, lower, upper = identify_outliers(data, column)
        num_outliers = len(outliers)
        percentage = round((num_outliers / len(data)) * 100, 2)
        summary.append({
            "Column": column,
            "Total Outliers": num_outliers,
            "Percentage (%)": percentage,
            "Lower Bound": lower,
            "Upper Bound": upper
        })
    return pd.DataFrame(summary)


def visualize_all(hour_df: pd.DataFrame):
    plot_features_distribution(hour_df)
    bike_rental_hour(hour_df)
    bike_rental_day(hour_df)
    bike_rental_month(hour_df)
    average_several_conditions(hour_df)
    numerical_features(hour_df)
    count_weekdays_weekends(hour_df)
    count_each_year(hour_df)
    seasonwise_distribution(hour_df)
    impact_of_holidays(hour_df)
    spearman_correlation(hour_df)
    outlier_detection(hour_df)

