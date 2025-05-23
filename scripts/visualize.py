#!/usr/bin/env python3
"""
Conway's Game of Life - Performance Visualization Script
Generates plots and analysis from performance benchmark results.
"""

import sys
from pathlib import Path
import argparse

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install with: pip install pandas matplotlib numpy")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("⚠️  Seaborn not available - using matplotlib defaults")
    HAS_SEABORN = False

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        print("⚠️  Using default matplotlib style (seaborn not available)")

if HAS_SEABORN:
    try:
        sns.set_palette("husl")
    except:
        print("⚠️  Could not set seaborn palette")


class PerformanceVisualizer:
    def __init__(self, results_file='performance_results.txt', output_dir='plots'):
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        """Load performance results from CSV file."""
        try:
            # Read the CSV file, skipping comment lines
            self.df = pd.read_csv(self.results_file, comment='#')
            print(f"✅ Loaded {len(self.df)} performance results")

            # Debug: Show actual columns
            print(f"📋 Available columns: {list(self.df.columns)}")

            # Check for required columns and try to fix common issues
            required_cols = ['Configuration', 'MPI_Processes',
                             'OpenMP_Threads', 'Total_Cores', 'Real_Time', 'Speedup']
            missing_cols = [
                col for col in required_cols if col not in self.df.columns]

            if missing_cols:
                print(f"⚠️  Missing expected columns: {missing_cols}")

                # Try to map common alternative column names
                column_mapping = {
                    'mpi_processes': 'MPI_Processes',
                    'openmp_threads': 'OpenMP_Threads',
                    'total_cores': 'Total_Cores',
                    'real_time': 'Real_Time',
                    'user_time': 'User_Time',
                    'sys_time': 'Sys_Time',
                    'configuration': 'Configuration'
                }

                # Apply column mapping
                for old_name, new_name in column_mapping.items():
                    if old_name in self.df.columns and new_name not in self.df.columns:
                        self.df.rename(
                            columns={old_name: new_name}, inplace=True)
                        print(f"🔄 Mapped column '{old_name}' to '{new_name}'")

                # Check again after mapping
                missing_cols = [
                    col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    print(f"❌ Still missing required columns: {missing_cols}")
                    print("📄 First few rows of data:")
                    print(self.df.head())
                    return False

            # Convert numeric columns to proper data types
            numeric_columns = ['MPI_Processes', 'OpenMP_Threads',
                               'Total_Cores', 'Real_Time', 'User_Time', 'Sys_Time', 'Speedup']
            for col in numeric_columns:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_numeric(
                            self.df[col], errors='coerce')
                        # Check for any NaN values that might indicate conversion issues
                        if self.df[col].isna().any():
                            print(f"⚠️  Warning: Some values in '{
                                  col}' could not be converted to numbers")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not convert column '{
                              col}' to numeric: {e}")

            print(f"✅ All required columns found")
            return True

        except FileNotFoundError:
            print(f"❌ Error: Results file '{self.results_file}' not found.")
            print(
                "   Please run the performance benchmark first: ./scripts/performance_test.sh")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print(f"📄 File content preview:")
            try:
                with open(self.results_file, 'r') as f:
                    lines = f.readlines()[:10]
                    for i, line in enumerate(lines, 1):
                        print(f"   Line {i}: {line.strip()}")
            except:
                print("   Could not read file")
            return False

    def create_speedup_plot(self):
        """Create speedup vs core count plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Separate data by parallelization type
        pure_openmp = self.df[self.df['MPI_Processes'] == 1]
        pure_mpi = self.df[self.df['OpenMP_Threads'] == 1]
        hybrid = self.df[(self.df['MPI_Processes'] > 1) &
                         (self.df['OpenMP_Threads'] > 1)]

        # Plot different categories
        if not pure_openmp.empty:
            ax.plot(pure_openmp['Total_Cores'], pure_openmp['Speedup'],
                    'o-', label='Pure OpenMP', linewidth=2, markersize=8)

        if not pure_mpi.empty:
            ax.plot(pure_mpi['Total_Cores'], pure_mpi['Speedup'],
                    's-', label='Pure MPI', linewidth=2, markersize=8)

        if not hybrid.empty:
            ax.plot(hybrid['Total_Cores'], hybrid['Speedup'],
                    '^-', label='Hybrid MPI+OpenMP', linewidth=2, markersize=8)

        # Add ideal speedup line
        max_cores = self.df['Total_Cores'].max()
        ideal_cores = np.arange(1, max_cores + 1)
        ax.plot(ideal_cores, ideal_cores, '--', color='gray',
                alpha=0.7, label='Ideal Speedup')

        # Formatting
        ax.set_xlabel('Number of Cores', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Game of Life - Parallel Speedup Analysis',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_cores + 0.5)
        ax.set_ylim(0.5, max(max_cores, self.df['Speedup'].max()) + 0.5)

        # Add annotations for best points
        best_speedup_idx = self.df['Speedup'].idxmax()
        best_row = self.df.iloc[best_speedup_idx]
        ax.annotate(f'Best: {best_row["Speedup"]:.2f}x\n({best_row["Configuration"]})',
                    xy=(best_row['Total_Cores'], best_row['Speedup']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        speedup_file = self.output_dir / 'speedup_analysis.png'
        plt.savefig(speedup_file, dpi=300, bbox_inches='tight')
        print(f"📊 Speedup plot saved: {speedup_file}")

        return fig

    def create_efficiency_plot(self):
        """Create parallel efficiency plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate efficiency (speedup / cores) - ensure numeric conversion
        self.df['Efficiency'] = pd.to_numeric(
            self.df['Speedup'], errors='coerce') / pd.to_numeric(self.df['Total_Cores'], errors='coerce') * 100

        # Remove any NaN values
        valid_data = self.df.dropna(subset=['Efficiency'])

        if valid_data.empty:
            print("⚠️  Warning: No valid efficiency data to plot")
            return fig

        # Create bar plot
        bars = ax.bar(range(len(valid_data)), valid_data['Efficiency'],
                      color=plt.cm.viridis(np.linspace(0, 1, len(valid_data))))

        # Add efficiency values on top of bars
        for i, (bar, efficiency) in enumerate(zip(bars, valid_data['Efficiency'])):
            if not pd.isna(efficiency):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{efficiency:.1f}%', ha='center', va='bottom', fontsize=10)

        # Formatting
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
        ax.set_title('Game of Life - Parallel Efficiency Analysis',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_data)))
        ax.set_xticklabels(
            valid_data['Configuration'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)

        # Add horizontal line at 100% efficiency
        ax.axhline(y=100, color='red', linestyle='--',
                   alpha=0.7, label='Perfect Efficiency')
        ax.legend()

        plt.tight_layout()
        efficiency_file = self.output_dir / 'efficiency_analysis.png'
        plt.savefig(efficiency_file, dpi=300, bbox_inches='tight')
        print(f"📊 Efficiency plot saved: {efficiency_file}")

        return fig

    def create_scaling_heatmap(self):
        """Create heatmap showing performance across MPI/OpenMP combinations."""
        # Create pivot table for heatmap
        pivot_data = self.df.pivot_table(values='Speedup',
                                         index='MPI_Processes',
                                         columns='OpenMP_Threads',
                                         fill_value=np.nan)

        fig, ax = plt.subplots(figsize=(10, 8))

        if HAS_SEABORN:
            # Create heatmap with seaborn
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd',
                        ax=ax, cbar_kws={'label': 'Speedup'})
        else:
            # Create heatmap with matplotlib
            im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')

            # Add text annotations
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center')

            # Set ticks and labels
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Speedup')

        ax.set_title('Game of Life - Scaling Heatmap\n(MPI Processes vs OpenMP Threads)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('OpenMP Threads per Process', fontsize=12)
        ax.set_ylabel('MPI Processes', fontsize=12)

        plt.tight_layout()
        heatmap_file = self.output_dir / 'scaling_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"📊 Scaling heatmap saved: {heatmap_file}")

        return fig

    def create_timing_breakdown(self):
        """Create timing breakdown chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Absolute execution times
        x_pos = np.arange(len(self.df))
        width = 0.35

        ax1.bar(x_pos - width/2, self.df['Real_Time'], width,
                label='Real Time', alpha=0.8)
        ax1.bar(x_pos + width/2, self.df['User_Time'], width,
                label='User Time', alpha=0.8)

        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Breakdown')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.df['Configuration'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Time vs cores scatter
        colors = plt.cm.plasma(np.linspace(0, 1, len(self.df)))
        scatter = ax2.scatter(self.df['Total_Cores'], self.df['Real_Time'],
                              c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add configuration labels
        for i, config in enumerate(self.df['Configuration']):
            ax2.annotate(config, (self.df['Total_Cores'].iloc[i], self.df['Real_Time'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

        ax2.set_xlabel('Number of Cores')
        ax2.set_ylabel('Real Time (seconds)')
        ax2.set_title('Performance vs Core Count')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        timing_file = self.output_dir / 'timing_breakdown.png'
        plt.savefig(timing_file, dpi=300, bbox_inches='tight')
        print(f"📊 Timing breakdown saved: {timing_file}")

        return fig

    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        report_file = self.output_dir / 'performance_report.md'

        with open(report_file, 'w') as f:
            f.write("# Conway's Game of Life - Performance Analysis Report\n\n")
            f.write(f"Generated on: {
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            f.write("## Performance Summary\n\n")
            best_speedup = self.df.loc[self.df['Speedup'].idxmax()]
            f.write(
                f"- **Best Speedup**: {best_speedup['Speedup']:.2f}x ({best_speedup['Configuration']})\n")
            f.write(
                f"- **Best Efficiency**: {(self.df['Speedup'] / self.df['Total_Cores'] * 100).max():.1f}%\n")
            f.write(f"- **Configurations Tested**: {len(self.df)}\n")
            f.write(
                f"- **Maximum Cores Used**: {self.df['Total_Cores'].max()}\n\n")

            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write(
                "| Configuration | MPI Processes | OpenMP Threads | Total Cores | Time (s) | Speedup | Efficiency |\n")
            f.write(
                "|---------------|---------------|----------------|-------------|----------|---------|------------|\n")

            for _, row in self.df.iterrows():
                try:
                    efficiency = float(row['Speedup']) / \
                        float(row['Total_Cores']) * 100
                    f.write(f"| {row['Configuration']} | {int(row['MPI_Processes'])} | {int(row['OpenMP_Threads'])} | "
                            f"{int(row['Total_Cores'])} | {float(row['Real_Time']):.2f} | {float(row['Speedup']):.2f}x | {efficiency:.1f}% |\n")
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Could not format row {
                          row['Configuration']}: {e}")
                    f.write(f"| {row['Configuration']} | {row['MPI_Processes']} | {row['OpenMP_Threads']} | "
                            f"{row['Total_Cores']} | {row['Real_Time']} | {row['Speedup']} | N/A |\n")

            # Analysis insights
            f.write("\n## Analysis Insights\n\n")

            # Find best approach for different core counts
            pure_openmp = self.df[self.df['MPI_Processes'] == 1]
            pure_mpi = self.df[self.df['OpenMP_Threads'] == 1]
            hybrid = self.df[(self.df['MPI_Processes'] > 1) &
                             (self.df['OpenMP_Threads'] > 1)]

            if not pure_openmp.empty:
                best_openmp = pure_openmp.loc[pure_openmp['Speedup'].idxmax()]
                f.write(f"- **Best Pure OpenMP**: {best_openmp['Speedup']:.2f}x with {
                        best_openmp['OpenMP_Threads']} threads\n")

            if not pure_mpi.empty:
                best_mpi = pure_mpi.loc[pure_mpi['Speedup'].idxmax()]
                f.write(f"- **Best Pure MPI**: {best_mpi['Speedup']:.2f}x with {
                        best_mpi['MPI_Processes']} processes\n")

            if not hybrid.empty:
                best_hybrid = hybrid.loc[hybrid['Speedup'].idxmax()]
                f.write(f"- **Best Hybrid**: {best_hybrid['Speedup']:.2f}x with {
                        best_hybrid['MPI_Processes']} processes × {best_hybrid['OpenMP_Threads']} threads\n")

            f.write(f"\n## Recommendations\n\n")
            f.write(f"Based on the performance analysis:\n\n")
            f.write(
                f"1. **Optimal Configuration**: {best_speedup['Configuration']}\n")
            f.write(f"2. **Scalability**: {'Good' if best_speedup['Speedup'] >
                    best_speedup['Total_Cores'] * 0.7 else 'Limited'} scaling observed\n")
            # Determine communication overhead
            if len(hybrid) > 0 and not pure_mpi.empty and hybrid['Speedup'].max() > pure_mpi['Speedup'].max():
                comm_overhead = "Low"
            elif len(hybrid) == 0 and len(pure_mpi) == 0:
                comm_overhead = "Unknown"
            else:
                comm_overhead = "Moderate"
            f.write(f"3. **Communication Overhead**: {comm_overhead}\n")

        print(f"📄 Performance report saved: {report_file}")

    def run_analysis(self):
        """Run complete performance analysis."""
        print("🧬 Conway's Game of Life - Performance Visualization")
        print("=" * 50)

        if not self.load_data():
            return False

        print(f"\n📊 Generating visualizations...")

        # Create all plots
        self.create_speedup_plot()
        self.create_efficiency_plot()
        self.create_scaling_heatmap()
        self.create_timing_breakdown()

        # Generate report
        self.generate_performance_report()

        print(f"\n✅ Analysis complete! Check the '{
              self.output_dir}' directory for results.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Game of Life performance results')
    parser.add_argument('-i', '--input', default='performance_results.txt',
                        help='Input results file (default: performance_results.txt)')
    parser.add_argument('-o', '--output', default='plots',
                        help='Output directory for plots (default: plots)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')

    args = parser.parse_args()

    # Create visualizer
    visualizer = PerformanceVisualizer(args.input, args.output)

    # Run analysis
    success = visualizer.run_analysis()

    if success and args.show:
        print("\n🖼️  Displaying plots...")
        plt.show()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
