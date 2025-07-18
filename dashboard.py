#!/usr/bin/env python3
"""
LLM Testing Results Dashboard
Automated analysis and visualization of LLM prompt injection testing results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
import glob
import os

warnings.filterwarnings('ignore')

class LLMTestingDashboard:
    """
    Comprehensive dashboard for analyzing LLM prompt injection testing results
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.summary_data = {}
        self.detailed_data = {}
        self.models = []
        
        # Set up styling
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load all summary and detailed data files"""
        print("📊 Loading LLM testing results...")
        
        # Load summary files
        summary_files = glob.glob(str(self.results_dir / "*_summary_*.json"))
        for file_path in summary_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_name = data['model'].replace('/', '_')
                self.summary_data[model_name] = data
                self.models.append(model_name)
                
        # Load detailed CSV files
        csv_files = glob.glob(str(self.results_dir / "*_detailed_*.csv"))
        for file_path in csv_files:
            model_name = Path(file_path).stem.replace('_detailed_' + file_path.split('_')[-1], '')
            try:
                df = pd.read_csv(file_path)
                self.detailed_data[model_name] = df
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                
        print(f"✅ Loaded data for {len(self.models)} models")
        
    def create_overview_metrics(self) -> Dict[str, Any]:
        """Calculate overview metrics across all models"""
        if not self.summary_data:
            return {}
            
        total_trials = sum(data['overall_stats']['total_trials'] for data in self.summary_data.values())
        total_poisoned = sum(data['overall_stats']['poisoned_selections'] for data in self.summary_data.values())
        avg_injection_rate = np.mean([data['overall_stats']['injection_success_rate'] for data in self.summary_data.values()])
        avg_robustness = np.mean([data['overall_stats']['model_robustness_score'] for data in self.summary_data.values()])
        avg_response_time = np.mean([data['overall_stats']['average_response_time'] for data in self.summary_data.values()])
        
        return {
            'total_models_tested': len(self.models),
            'total_trials': total_trials,
            'total_poisoned_selections': total_poisoned,
            'average_injection_success_rate': avg_injection_rate,
            'average_robustness_score': avg_robustness,
            'average_response_time': avg_response_time
        }
        
    def plot_model_robustness_comparison(self):
        """Create horizontal bar chart comparing model robustness scores"""
        models = []
        robustness_scores = []
        injection_rates = []
        
        for model, data in self.summary_data.items():
            models.append(model.replace('_', '/'))
            robustness_scores.append(data['overall_stats']['model_robustness_score'])
            injection_rates.append(data['overall_stats']['injection_success_rate'])
            
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add robustness scores
        fig.add_trace(
            go.Bar(
                y=models,
                x=robustness_scores,
                name='Robustness Score (%)',
                orientation='h',
                marker_color='lightblue',
                text=[f'{score:.1f}%' for score in robustness_scores],
                textposition='inside'
            ),
            secondary_y=False,
        )
        
        # Add injection success rates as scatter
        fig.add_trace(
            go.Scatter(
                y=models,
                x=injection_rates,
                mode='markers+text',
                name='Injection Success Rate (%)',
                marker=dict(color='red', size=10, symbol='diamond'),
                text=[f'{rate:.1f}%' for rate in injection_rates],
                textposition='middle right'
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title='🛡️ Model Robustness vs Injection Success Rate',
            height=600,
            showlegend=True,
            font=dict(size=12)
        )
        fig.update_xaxes(title_text="Score (%)")
        fig.update_yaxes(title_text="Models")
        
        return fig
        
    def plot_strategy_effectiveness(self):
        """Analyze effectiveness of different injection strategies across models"""
        strategy_data = {}
        
        for model, data in self.summary_data.items():
            for var_key, var_data in data['variation_rankings'].items():
                strategy = var_data['strategy_name']
                if strategy not in strategy_data:
                    strategy_data[strategy] = {
                        'success_rates': [],
                        'models': [],
                        'variation_numbers': []
                    }
                strategy_data[strategy]['success_rates'].append(var_data['success_rate'])
                strategy_data[strategy]['models'].append(model.replace('_', '/'))
                strategy_data[strategy]['variation_numbers'].append(var_data['variation_number'])
        
        # Create box plot
        strategies = []
        success_rates = []
        models = []
        
        for strategy, data in strategy_data.items():
            strategies.extend([strategy] * len(data['success_rates']))
            success_rates.extend(data['success_rates'])
            models.extend(data['models'])
            
        df_strategies = pd.DataFrame({
            'Strategy': strategies,
            'Success Rate': success_rates,
            'Model': models
        })
        
        fig = px.box(
            df_strategies,
            x='Strategy',
            y='Success Rate',
            color='Strategy',
            title='📈 Strategy Effectiveness Across All Models',
            labels={'Success Rate': 'Success Rate (%)'},
            height=600
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(showlegend=False, font=dict(size=12))
        
        return fig
        
    def plot_response_time_analysis(self):
        """Analyze response times across models and strategies"""
        models = []
        response_times = []
        
        for model, data in self.summary_data.items():
            models.append(model.replace('_', '/'))
            response_times.append(data['overall_stats']['average_response_time'])
            
        fig = go.Figure(data=go.Bar(
            x=models,
            y=response_times,
            text=[f'{time:.2f}s' for time in response_times],
            textposition='auto',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='⏱️ Average Response Time by Model',
            xaxis_title='Models',
            yaxis_title='Response Time (seconds)',
            height=500,
            font=dict(size=12)
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    def plot_temperature_impact(self):
        """Analyze the impact of temperature on injection success"""
        if not self.detailed_data:
            return None
            
        # Combine all detailed data
        all_data = []
        for model, df in self.detailed_data.items():
            df_copy = df.copy()
            df_copy['model'] = model.replace('_', '/')
            all_data.append(df_copy)
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Group by temperature and calculate success rate
        temp_analysis = combined_df.groupby(['temperature', 'model']).agg({
            'is_poisoned_selected': ['count', 'sum']
        }).reset_index()
        
        temp_analysis.columns = ['temperature', 'model', 'total_trials', 'successful_injections']
        temp_analysis['success_rate'] = (temp_analysis['successful_injections'] / temp_analysis['total_trials']) * 100
        
        fig = px.line(
            temp_analysis,
            x='temperature',
            y='success_rate',
            color='model',
            title='🌡️ Temperature Impact on Injection Success Rate',
            labels={'success_rate': 'Success Rate (%)', 'temperature': 'Temperature'},
            markers=True,
            height=500
        )
        
        fig.update_layout(font=dict(size=12))
        
        return fig
        
    def plot_variation_heatmap(self):
        """Create heatmap showing success rates by variation and model"""
        # Prepare data for heatmap
        models = []
        variations = []
        success_rates = []
        
        for model, data in self.summary_data.items():
            for var_key, var_data in data['variation_rankings'].items():
                models.append(model.replace('_', '/'))
                variations.append(f"Var {var_data['variation_number']}: {var_data['strategy_name']}")
                success_rates.append(var_data['success_rate'])
                
        df_heatmap = pd.DataFrame({
            'Model': models,
            'Variation': variations,
            'Success Rate': success_rates
        })
        
        # Pivot for heatmap
        heatmap_data = df_heatmap.pivot(index='Model', columns='Variation', values='Success Rate')
        
        fig = px.imshow(
            heatmap_data,
            title='🔥 Success Rate Heatmap: Models vs Injection Strategies',
            labels=dict(x="Injection Strategy", y="Model", color="Success Rate (%)"),
            aspect="auto",
            color_continuous_scale='RdYlGn_r',
            height=800
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(font=dict(size=10))
        
        return fig
        
    def create_summary_table(self):
        """Create a summary table of all models"""
        summary_rows = []
        
        for model, data in self.summary_data.items():
            summary_rows.append({
                'Model': model.replace('_', '/'),
                'Total Trials': data['overall_stats']['total_trials'],
                'Poisoned Selections': data['overall_stats']['poisoned_selections'],
                'Injection Success Rate (%)': f"{data['overall_stats']['injection_success_rate']:.1f}",
                'Robustness Score (%)': f"{data['overall_stats']['model_robustness_score']:.1f}",
                'Avg Response Time (s)': f"{data['overall_stats']['average_response_time']:.2f}"
            })
            
        df_summary = pd.DataFrame(summary_rows)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_summary.columns),
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df_summary[col] for col in df_summary.columns],
                fill_color='white',
                align='center',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title='📋 Model Performance Summary',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    def generate_dashboard(self, output_file: str = "llm_testing_dashboard.html"):
        """Generate complete interactive dashboard"""
        print("🚀 Generating comprehensive dashboard...")
        
        # Calculate overview metrics
        overview = self.create_overview_metrics()
        
        # Create all plots
        plots = {
            'robustness': self.plot_model_robustness_comparison(),
            'strategy': self.plot_strategy_effectiveness(),
            'response_time': self.plot_response_time_analysis(),
            'temperature': self.plot_temperature_impact(),
            'heatmap': self.plot_variation_heatmap(),
            'summary': self.create_summary_table()
        }
        
        # Create HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Testing Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            margin-top: 10px;
        }}
        .plot-container {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 LLM Prompt Injection Testing Dashboard</h1>
        <p>Comprehensive Analysis of Model Robustness and Injection Strategy Effectiveness</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{overview.get('total_models_tested', 0)}</div>
            <div class="metric-label">Models Tested</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overview.get('total_trials', 0):,}</div>
            <div class="metric-label">Total Trials</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overview.get('average_injection_success_rate', 0):.1f}%</div>
            <div class="metric-label">Avg Injection Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overview.get('average_robustness_score', 0):.1f}%</div>
            <div class="metric-label">Avg Robustness</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overview.get('average_response_time', 0):.2f}s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
    </div>
"""
        
        # Add plots to HTML
        plot_htmls = {}
        for name, fig in plots.items():
            if fig is not None:
                plot_htmls[name] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
                
        html_content += f"""
    <div class="plot-container">
        {plot_htmls.get('summary', '')}
    </div>
    
    <div class="plot-container">
        {plot_htmls.get('robustness', '')}
    </div>
    
    <div class="plot-container">
        {plot_htmls.get('strategy', '')}
    </div>
    
    <div class="plot-container">
        {plot_htmls.get('heatmap', '')}
    </div>
    
    <div class="plot-container">
        {plot_htmls.get('response_time', '')}
    </div>
"""
        
        if plot_htmls.get('temperature'):
            html_content += f"""
    <div class="plot-container">
        {plot_htmls.get('temperature', '')}
    </div>
"""
        
        html_content += """
    <div class="footer">
        <p>Dashboard generated by LLM Testing Analysis Tool</p>
        <p>Data represents comprehensive prompt injection testing across multiple models and strategies</p>
    </div>
</body>
</html>
"""
        
        # Save dashboard
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Dashboard saved as: {output_file}")
        return output_file
        
    def generate_static_plots(self, output_dir: str = "dashboard_plots"):
        """Generate static matplotlib plots as backup"""
        print("📊 Generating static plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Model comparison plot
        plt.figure(figsize=(12, 8))
        models = [model.replace('_', '/') for model in self.summary_data.keys()]
        robustness_scores = [data['overall_stats']['model_robustness_score'] for data in self.summary_data.values()]
        injection_rates = [data['overall_stats']['injection_success_rate'] for data in self.summary_data.values()]
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Robustness scores
        color = 'tab:blue'
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Robustness Score (%)', color=color)
        bars = ax1.bar(models, robustness_scores, color=color, alpha=0.7, label='Robustness Score')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, robustness_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}%', ha='center', va='bottom')
        
        # Injection rates on secondary axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Injection Success Rate (%)', color=color)
        ax2.plot(models, injection_rates, color=color, marker='o', linewidth=2, markersize=8, label='Injection Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Model Robustness vs Injection Success Rate', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Static plots saved in: {output_dir}")

def main():
    """Main function to run the dashboard generation"""
    parser = argparse.ArgumentParser(description='Generate LLM Testing Results Dashboard')
    parser.add_argument('--results-dir', default='results', help='Directory containing result files')
    parser.add_argument('--output', default='llm_testing_dashboard.html', help='Output HTML file')
    parser.add_argument('--static-plots', action='store_true', help='Also generate static plots')
    
    args = parser.parse_args()
    
    print("🔬 LLM Testing Results Dashboard Generator")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = LLMTestingDashboard(args.results_dir)
    
    # Load data
    dashboard.load_data()
    
    if not dashboard.summary_data:
        print("❌ No data found. Please check the results directory.")
        return
    
    # Generate dashboard
    output_file = dashboard.generate_dashboard(args.output)
    
    # Generate static plots if requested
    if args.static_plots:
        dashboard.generate_static_plots()
    
    print("\n🎉 Dashboard generation complete!")
    print(f"📊 Open {output_file} in your browser to view the interactive dashboard")
    
    # Show summary statistics
    overview = dashboard.create_overview_metrics()
    print("\n📈 Quick Summary:")
    print(f"   • Models tested: {overview.get('total_models_tested', 0)}")
    print(f"   • Total trials: {overview.get('total_trials', 0):,}")
    print(f"   • Average injection success rate: {overview.get('average_injection_success_rate', 0):.1f}%")
    print(f"   • Average model robustness: {overview.get('average_robustness_score', 0):.1f}%")

if __name__ == "__main__":
    main()
