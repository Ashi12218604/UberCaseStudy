import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')  # Updated style name for newer matplotlib versions
    sns.set_theme(style="whitegrid")  # Use seaborn's theming
    sns.set_palette("viridis")
except:
    # Fallback to matplotlib's default style if seaborn is not available
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    print("Note: Using default matplotlib style as seaborn style is not available")

class TrafficPatternAnalyzer:
    def __init__(self, data_path, output_dir='results/peak_analysis'):
        self.data_path = data_path
        self.output_dir = output_dir
        self._create_output_dir()
        self.df = None
        
    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the traffic data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, parse_dates=['DateTime'])
        
        # Set DateTime as index
        self.df.set_index('DateTime', inplace=True)
        
        # Extract time-based features
        self.df['Hour'] = self.df.index.hour
        self.df['DayOfWeek'] = self.df.index.dayofweek
        self.df['Month'] = self.df.index.month
        self.df['DayOfMonth'] = self.df.index.day
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        
        print(f"Data loaded with {len(self.df)} records")
        return self.df
    
    def calculate_congestion_metrics(self):
        """Calculate congestion metrics by hour"""
        print("Calculating congestion metrics...")
        
        # Group by hour and calculate metrics
        hourly_metrics = self.df.groupby('Hour').agg({
            'Vehicles': ['mean', 'median', 'std', 'count'],
            'Avg_Temp': 'mean',
            'Precipitation': 'mean',
            'Event_Count': 'sum'
        })
        
        # Flatten multi-index columns
        hourly_metrics.columns = ['_'.join(col).strip() for col in hourly_metrics.columns.values]
        hourly_metrics.rename(columns={
            'Vehicles_mean': 'Avg_Vehicles',
            'Vehicles_median': 'Median_Vehicles',
            'Vehicles_std': 'Std_Vehicles',
            'Vehicles_count': 'Count',
            'Avg_Temp_mean': 'Avg_Temperature',
            'Precipitation_mean': 'Avg_Precipitation',
            'Event_Count_sum': 'Total_Events'
        }, inplace=True)
        
        # Calculate peak hour thresholds (1 standard deviation above mean)
        mean_vehicles = hourly_metrics['Avg_Vehicles'].mean()
        std_vehicles = hourly_metrics['Avg_Vehicles'].std()
        hourly_metrics['Is_Peak'] = hourly_metrics['Avg_Vehicles'] > (mean_vehicles + std_vehicles)
        
        return hourly_metrics
    
    def analyze_weekly_patterns(self):
        """Analyze traffic patterns by day of week"""
        print("Analyzing weekly patterns...")
        
        # Group by day of week and hour
        weekly_patterns = self.df.groupby(['DayOfWeek', 'Hour'])['Vehicles'].mean().unstack(level=0)
        
        # Rename days for better readability
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_patterns.columns = [day_names[i] for i in weekly_patterns.columns]
        
        return weekly_patterns
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal traffic patterns"""
        print("Analyzing seasonal patterns...")
        
        # Define seasons (Northern Hemisphere)
        season_mapping = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        
        self.df['Season'] = self.df['Month'].map(season_mapping)
        
        # Group by season and hour
        seasonal_patterns = self.df.groupby(['Season', 'Hour'])['Vehicles'].mean().unstack(level=0)
        
        return seasonal_patterns
    
    def plot_hourly_patterns(self, hourly_metrics):
        """Plot hourly traffic patterns"""
        plt.figure(figsize=(14, 7))
        
        # Plot average vehicles by hour
        ax = sns.lineplot(data=hourly_metrics, x=hourly_metrics.index, y='Avg_Vehicles', 
                         marker='o', linewidth=2.5, color='#3498db')
        
        # Highlight peak hours
        peak_hours = hourly_metrics[hourly_metrics['Is_Peak']]
        plt.scatter(peak_hours.index, peak_hours['Avg_Vehicles'], 
                   color='#e74c3c', s=100, zorder=5, 
                   label='Peak Hours')
        
        # Add mean line
        mean_vehicles = hourly_metrics['Avg_Vehicles'].mean()
        plt.axhline(y=mean_vehicles, color='#2ecc71', linestyle='--', 
                   label=f'Average: {mean_vehicles:.1f} vehicles')
        
        # Customize plot
        plt.title('Average Hourly Traffic Volume', fontsize=16, pad=20)
        plt.xlabel('Hour of Day', fontsize=12, labelpad=10)
        plt.ylabel('Average Number of Vehicles', fontsize=12, labelpad=10)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hourly_traffic_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weekly_patterns(self, weekly_patterns):
        """Plot weekly traffic patterns"""
        plt.figure(figsize=(16, 8))
        
        # Plot each day's pattern
        for day in weekly_patterns.columns:
            sns.lineplot(data=weekly_patterns, x=weekly_patterns.index, y=day, 
                        marker='o', label=day, linewidth=2)
        
        # Customize plot
        plt.title('Average Hourly Traffic by Day of Week', fontsize=16, pad=20)
        plt.xlabel('Hour of Day', fontsize=12, labelpad=10)
        plt.ylabel('Average Number of Vehicles', fontsize=12, labelpad=10)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/weekly_traffic_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_seasonal_patterns(self, seasonal_patterns):
        """Plot seasonal traffic patterns"""
        # Reorder seasons for logical x-axis
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_patterns = seasonal_patterns[season_order]
        
        plt.figure(figsize=(16, 7))
        
        # Plot each season's pattern
        for season in seasonal_patterns.columns:
            sns.lineplot(data=seasonal_patterns, x=seasonal_patterns.index, y=season, 
                        marker='o', label=season, linewidth=2.5)
        
        # Customize plot
        plt.title('Average Hourly Traffic by Season', fontsize=16, pad=20)
        plt.xlabel('Hour of Day', fontsize=12, labelpad=10)
        plt.ylabel('Average Number of Vehicles', fontsize=12, labelpad=10)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/seasonal_traffic_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def correlate_with_weather(self):
        """Analyze correlation between traffic and weather conditions"""
        print("Analyzing weather correlations...")
        
        # Calculate daily aggregates
        daily_data = self.df.resample('D').agg({
            'Vehicles': 'sum',
            'Avg_Temp': 'mean',
            'Precipitation': 'mean',
            'Wind_Speed': 'mean',
            'Event_Count': 'sum'
        })
        
        # Calculate correlation matrix
        corr_matrix = daily_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
        plt.title('Correlation Between Traffic and Weather/Events', fontsize=16, pad=20)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/weather_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting traffic pattern analysis...\n")
        
        # Load and preprocess data
        self.load_data()
        
        # Calculate congestion metrics
        hourly_metrics = self.calculate_congestion_metrics()
        
        # Analyze weekly patterns
        weekly_patterns = self.analyze_weekly_patterns()
        
        # Analyze seasonal patterns
        seasonal_patterns = self.analyze_seasonal_patterns()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_hourly_patterns(hourly_metrics)
        self.plot_weekly_patterns(weekly_patterns)
        self.plot_seasonal_patterns(seasonal_patterns)
        
        # Analyze weather correlations
        weather_corr = self.correlate_with_weather()
        
        # Save results to CSV
        hourly_metrics.to_csv(f"{self.output_dir}/hourly_metrics.csv")
        weekly_patterns.to_csv(f"{self.output_dir}/weekly_patterns.csv")
        seasonal_patterns.to_csv(f"{self.output_dir}/seasonal_patterns.csv")
        weather_corr.to_csv(f"{self.output_dir}/weather_correlations.csv")
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # Print summary of findings
        peak_hours = hourly_metrics[hourly_metrics['Is_Peak']].index.tolist()
        print(f"\nPeak Traffic Hours: {sorted(peak_hours)}")
        
        # Find busiest day of week
        busiest_day_idx = weekly_patterns.mean().idxmax()
        print(f"Busiest day of week: {busiest_day_idx}")
        
        # Find busiest season
        busiest_season = seasonal_patterns.mean().idxmax()
        print(f"Busiest season: {busiest_season}")


if __name__ == "__main__":
    # Initialize analyzer with path to integrated data
    analyzer = TrafficPatternAnalyzer(
        data_path='integrated_traffic_data.csv',
        output_dir='results/peak_analysis'
    )
    
    # Run the analysis
    analyzer.run_analysis()
