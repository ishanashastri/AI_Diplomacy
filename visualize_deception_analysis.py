import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class DiplomacyVisualization:
    def __init__(self, results):
        self.results = results
        self.overall = results['overall_metrics']
        self.patterns = results['strategic_patterns']
        self.games = results['game_metrics']
        self.tracking = results['all_tracking']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_betrayal_hierarchy_chart(self):
        """Horizontal bar chart showing betrayal rates by promise type"""
        hierarchy = self.patterns['betrayal_hierarchy']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(hierarchy.keys())
        rates = [hierarchy[t] * 100 for t in types]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.barh(types, rates, color=colors[:len(types)])
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.1f}%', va='center', fontweight='bold')
        
        ax.set_xlabel('Betrayal Rate (%)', fontsize=12)
        ax.set_title('Strategic Deception Hierarchy\n(Which Promises Are Most Likely to Be Broken) - Mistral Small (n=8)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(rates) * 1.2)
        
        plt.tight_layout()
        return fig
    
    def create_temporal_betrayal_heatmap(self):
        """Heatmap showing betrayal patterns across games and turns"""
        # Create matrix of betrayals by game and turn
        betrayal_matrix = defaultdict(lambda: defaultdict(int))
        
        import json
        game_max_turns = {}
        for track in self.tracking:
            # if not track.kept and track.broken_phase is not None:
            if not track.kept and track.broken_turn is not None:
                game_id = int(track.promise.promise_id.split('_')[0][1:])  # Extract game ID
                phase_to_ind= {}
                with open(f"./results/sam-exp081-bench/runs/run_{game_id:05d}/lmvsgame.json") as f:
                    g = json.load(f)
                    for phase_ind, phase in enumerate(g['phases']):
                        phase_to_ind[phase['name']] = phase_ind
                    game_max_turns[game_id] = len(g['phases'])-1
                # betrayal_matrix[game_id][phase_to_ind[track.broken_phase]] += 1
                betrayal_matrix[game_id][track.broken_turn] += 1
        
        # Convert to DataFrame
        max_turn = max(max(turns.keys()) if turns else [0] for turns in betrayal_matrix.values())
        
        data = []
        for game_id in range(len(self.games)):
            for turn in range(max_turn + 1):
                if turn <= game_max_turns[game_id]:
                    betrayals = betrayal_matrix[game_id][turn]
                else:
                    betrayals = -1
                
                data.append({
                    'Game': game_id,
                    'Turn': turn,
                    'Betrayals': betrayals
                })
        
        df = pd.DataFrame(data)
        pivot = df.pivot(index='Game', columns='Turn', values='Betrayals').fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='g', cmap='Reds', cbar_kws={'label': 'Betrayals'})
        ax.set_title('Betrayal Patterns Across Games and Turns - Mistral Small (n=8)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Game ID', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_game_comparison_violin(self):
        """Violin plot comparing betrayal rates across games"""
        game_betrayal_rates = [g['betrayal_rate'] for g in self.games]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        violin_parts = ax.violinplot([game_betrayal_rates], positions=[0], widths=0.6)
        
        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('#FF6B6B')
            pc.set_alpha(0.7)
        
        # Add individual points
        y_pos = np.random.normal(0, 0.04, len(game_betrayal_rates))
        ax.scatter(y_pos, game_betrayal_rates, alpha=0.6, s=50, color='darkred')
        
        ax.set_ylabel('Betrayal Rate', fontsize=12)
        ax.set_title('Betrayal Rate Distribution Across Games - Mistral Small (n=8)', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        
        # Add statistics text
        mean_rate = np.mean(game_betrayal_rates)
        std_rate = np.std(game_betrayal_rates)
        ax.text(0.02, 0.95, f'Mean: {mean_rate:.2%}\nStd: {std_rate:.2%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_target_analysis_pie(self):
        """Pie chart showing which countries are betrayed most"""
        targets = self.patterns['target_preferences']
        
        if not targets:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        countries = list(targets.keys())
        betrayals = list(targets.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
        
        wedges, texts, autotexts = ax.pie(betrayals, labels=countries, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.legend()
        ax.set_title('Target Preferences for Betrayal\n(Which Countries Get Betrayed Most) - Mistral Small (n=8)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_time_to_betrayal_distribution(self):
        """Histogram of time-to-betrayal distribution"""
        betrayal_times = [t.time_to_betrayal for t in self.tracking 
                         if not t.kept and t.time_to_betrayal is not None]
        
        if not betrayal_times:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(betrayal_times, bins=max(betrayal_times) + 1, 
                                  alpha=0.7, color='#FF6B6B', edgecolor='black')
        
        # Color bars by frequency
        for i, (patch, freq) in enumerate(zip(patches, n)):
            patch.set_facecolor(plt.cm.Reds(freq / max(n)))
        
        ax.set_xlabel('Turns Until Betrayal', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Time to Betrayal - Mistral Small (n=8)', fontsize=14, fontweight='bold')
        
        # Add statistics
        mean_time = np.mean(betrayal_times)
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_time:.1f} turns')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_interactive_promise_timeline(self):
        """Interactive timeline showing promises and betrayals"""
        # Prepare data
        timeline_data = []
        
        for track in self.tracking:
            timeline_data.append({
                'Promise_ID': track.promise.promise_id,
                'Type': track.promise.promise_type,
                'Turn': track.promise.turn,
                'Kept': 'Kept' if track.kept else 'Broken',
                'Time_to_Betrayal': track.time_to_betrayal or 0,
                'Target': track.promise.receiver,
                'Game': int(track.promise.promise_id.split('_')[0][1:])
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create interactive scatter plot
        fig = px.scatter(df, x='Turn', y='Game', 
                        color='Type', symbol='Kept',
                        hover_data=['Target', 'Time_to_Betrayal'],
                        title='Promise Timeline Across All Games - Mistral Small (n=8)',
                        labels={'Turn': 'Game Turn', 'Game': 'Game ID'})
        
        fig.update_layout(height=600)
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        # Set up the subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Betrayal Hierarchy', 'Betrayal Rate Distribution',
                           'Target Preferences', 'Time to Betrayal',
                           'Game Comparison', 'Promise Type Distribution'),
            specs=[[{"type": "bar"}, {"type": "violin"}],
                   [{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Betrayal Hierarchy
        hierarchy = self.patterns['betrayal_hierarchy']
        fig.add_trace(go.Bar(
            x=list(hierarchy.values()),
            y=list(hierarchy.keys()),
            orientation='h',
            marker_color='#FF6B6B',
            name='Betrayal Rate'
        ), row=1, col=1)
        
        # 2. Target Preferences
        targets = self.patterns['target_preferences']
        if targets:
            fig.add_trace(go.Pie(
                labels=list(targets.keys()),
                values=list(targets.values()),
                name='Betrayal Targets'
            ), row=2, col=1)
        
        # 3. Time to Betrayal
        betrayal_times = [t.time_to_betrayal for t in self.tracking 
                         if not t.kept and t.time_to_betrayal is not None]
        if betrayal_times:
            fig.add_trace(go.Histogram(
                x=betrayal_times,
                marker_color='#4ECDC4',
                name='Time to Betrayal'
            ), row=2, col=2)
        
        # 4. Game Comparison
        game_betrayal_rates = [g['betrayal_rate'] for g in self.games]
        fig.add_trace(go.Scatter(
            x=list(range(len(game_betrayal_rates))),
            y=game_betrayal_rates,
            mode='markers+lines',
            marker_color='#45B7D1',
            name='Betrayal Rate by Game'
        ), row=3, col=1)
        
        # 5. Promise Type Distribution
        type_counts = dict(sorted(self.overall['promise_type_distribution'].items()))
        if 'conditional' in type_counts: del type_counts['conditional']
        fig.add_trace(go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            marker_color='#96CEB4',
            name='Promise Types'
        ), row=3, col=2)
        
        fig.update_layout(height=1200, title_text="Diplomacy Deception Analysis Dashboard - Mistral Small (n=8)")
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations and return them"""
        visualizations = {}
        
        print("Generating visualizations...")
        
        # Static plots
        visualizations['betrayal_hierarchy'] = self.create_betrayal_hierarchy_chart()
        visualizations['temporal_heatmap'] = self.create_temporal_betrayal_heatmap()
        visualizations['game_comparison'] = self.create_game_comparison_violin()
        visualizations['target_analysis'] = self.create_target_analysis_pie()
        visualizations['time_distribution'] = self.create_time_to_betrayal_distribution()
        
        # Interactive plots
        visualizations['interactive_timeline'] = self.create_interactive_promise_timeline()
        visualizations['dashboard'] = self.create_comprehensive_dashboard()
        
        return visualizations

# Usage function
def visualize_deception_analysis(results):
    """
    Create all visualizations for deception analysis results
    
    Args:
        results: Output from DiplomacyDeceptionAnalyzer
    """
    visualizer = DiplomacyVisualization(results)
    plots = visualizer.generate_all_visualizations()
    
    # Display static plots
    for name, plot in plots.items():
        if name not in ['interactive_timeline', 'dashboard'] and plot is not None:
            plt.figure(plot.number)
            plt.show()
    
    # Display interactive plots
    if plots['interactive_timeline']:
        plots['interactive_timeline'].show()
    
    if plots['dashboard']:
        plots['dashboard'].show()
    
    return plots
