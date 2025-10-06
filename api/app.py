"""
Simple Web UI for Factor Cache Query and Visualization
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path BEFORE local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils

# Local imports
from api.factor_cache_manager import FactorCacheManager
from api.factor_eval_client import FactorEvalClient

app = Flask(__name__)
CORS(app)

# Initialize cache manager
# Use absolute path to ensure correct resolution
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache_data"))
cache_manager = FactorCacheManager(cache_dir=cache_dir)

# Initialize API client
api_client = FactorEvalClient(base_url="http://localhost:9889")


@app.route('/')
def index():
    """Main page with query interface."""
    return render_template('index.html')


@app.route('/api/search_cache', methods=['POST'])
def search_cache():
    """Search for cached factor results."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    # Check if we have cache for this query
    if cache_manager.has_cache(expr, market, start_date, end_date):
        result = cache_manager.get_cache(expr, market, start_date, end_date)
        if result:
            # Manually update access count since get_cache already does this
            return jsonify({
                'success': True,
                'data': result,
                'from_cache': True
            })
    
    return jsonify({
        'success': False,
        'message': 'No cache found for this query'
    })


@app.route('/api/evaluate_factor', methods=['POST'])
def evaluate_factor():
    """Evaluate a factor expression (using API or direct computation)."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    use_cache = data.get('use_cache', True)
    
    # Try to evaluate via API first
    try:
        result = api_client.evaluate_factor(
            expr=expr,
            market=market,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': result
            })
    except Exception as e:
        print(f"API evaluation failed: {e}")
    
    return jsonify({
        'success': False,
        'message': 'Evaluation failed. Please check the API server is running.'
    })


@app.route('/api/get_cached_expressions', methods=['GET'])
def get_cached_expressions():
    """Get list of cached expressions."""
    limit = request.args.get('limit', 100, type=int)
    order_by = request.args.get('order_by', 'last_accessed')
    
    expressions = cache_manager.get_cached_expressions(limit=limit, order_by=order_by)
    
    # Log for debugging
    print(f"Found {len(expressions)} cached expressions")
    if expressions:
        print(f"Sample: {expressions[0]['expression'][:50]}... with {expressions[0]['access_count']} accesses")
    
    return jsonify({
        'success': True,
        'expressions': expressions
    })


@app.route('/api/get_daily_chart', methods=['POST'])
def get_daily_chart():
    """Generate daily IC/RankIC chart data."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    result = cache_manager.get_cache(expr, market, start_date, end_date)
    
    if not result or 'daily_metrics' not in result:
        return jsonify({
            'success': False,
            'message': 'No daily data available'
        })
    
    df = pd.DataFrame(result['daily_metrics'])
    
    # Create plotly chart
    fig = go.Figure()
    
    # Add IC line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ic'],
        mode='lines+markers',
        name='IC',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Add Rank IC line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rank_ic'],
        mode='lines+markers',
        name='Rank IC',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Daily IC and Rank IC - {market.upper()}',
        xaxis_title='Date',
        yaxis_title='Correlation',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'success': True,
        'chart': chart_json
    })


@app.route('/api/get_performance_stats', methods=['POST'])
def get_performance_stats():
    """Get performance statistics for a factor."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    result = cache_manager.get_cache(expr, market, start_date, end_date)
    
    if not result:
        return jsonify({
            'success': False,
            'message': 'No data available'
        })
    
    metrics = result.get('metrics', {})
    daily_metrics = result.get('daily_metrics', [])
    
    # Calculate additional statistics
    stats = {
        'mean_ic': metrics.get('ic', 0),
        'mean_rank_ic': metrics.get('rank_ic', 0),
        'icir': metrics.get('icir', 0),
        'rank_icir': metrics.get('rank_icir', 0),
        'n_dates': metrics.get('n_dates', 0)
    }
    
    if daily_metrics:
        df = pd.DataFrame(daily_metrics)
        ic_values = df['ic'].dropna()
        rank_ic_values = df['rank_ic'].dropna()
        
        if len(ic_values) > 0:
            stats['ic_std'] = float(ic_values.std())
            stats['ic_min'] = float(ic_values.min())
            stats['ic_max'] = float(ic_values.max())
            stats['ic_positive_ratio'] = float((ic_values > 0).mean())
        
        if len(rank_ic_values) > 0:
            stats['rank_ic_std'] = float(rank_ic_values.std())
            stats['rank_ic_min'] = float(rank_ic_values.min())
            stats['rank_ic_max'] = float(rank_ic_values.max())
            stats['rank_ic_positive_ratio'] = float((rank_ic_values > 0).mean())
    
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/api/cache_stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    stats = cache_manager.get_cache_stats()
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear cache entries."""
    data = request.json
    older_than_days = data.get('older_than_days')
    
    try:
        cache_manager.clear_cache(older_than_days=older_than_days)
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })


@app.route('/api/compare_factors', methods=['POST'])
def compare_factors():
    """Compare multiple factors."""
    data = request.json
    expressions = data.get('expressions', [])
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    if len(expressions) < 2:
        return jsonify({
            'success': False,
            'message': 'Need at least 2 expressions to compare'
        })
    
    # Use API client to compare factors
    try:
        response = api_client.api_request('/compare_factors', {
            'factors': expressions,
            'market': market,
            'start': start_date,
            'end': end_date
        })
        
        # Check if response is None (API request failed)
        if response is None:
            return jsonify({
                'success': False,
                'message': 'API request failed. Please check if the API server is running.'
            })
        
        if response.get('success'):
            return jsonify({
                'success': True,
                'comparison': response.get('comparison', [])
            })
        else:
            return jsonify({
                'success': False,
                'message': response.get('message', 'Comparison failed')
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })


@app.route('/api/get_factor_history', methods=['POST'])
def get_factor_history():
    """Get evaluation history for a factor."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market')
    
    if not expr:
        return jsonify({
            'success': False,
            'message': 'Expression required'
        })
    
    history = cache_manager.get_factor_history(expr, market)
    
    return jsonify({
        'success': True,
        'history': history
    })


@app.route('/api/export_data', methods=['POST'])
def export_data():
    """Export factor data."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    export_format = data.get('format', 'csv')
    
    try:
        response = api_client.api_request('/export_data', {
            'expr': expr,
            'market': market,
            'start': start_date,
            'end': end_date,
            'format': export_format
        })
        
        # Check if response is None (API request failed)
        if response is None:
            return jsonify({
                'success': False,
                'message': 'API request failed. Please check if the API server is running.'
            })
        
        if response.get('success'):
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'message': response.get('message', 'Export failed')
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })


@app.route('/api/get_top_factors', methods=['GET'])
def get_top_factors():
    """Get top performing factors based on IC or Rank IC."""
    limit = request.args.get('limit', 10, type=int)
    metric = request.args.get('metric', 'rank_ic')  # rank_ic or ic
    
    # Get all cached expressions with their performance data
    expressions = cache_manager.get_cached_expressions(limit=100)
    
    # Fetch metrics for each expression
    factors_with_metrics = []
    for expr in expressions:
        # Get the most recent evaluation - try all markets if no specific market data
        cache_data = cache_manager.get_latest_evaluation(expr['expression'])
        
        # If no data for default market, try common markets
        if not cache_data:
            for market in ['csi300', 'csi500', 'csi1000']:
                cache_data = cache_manager.get_latest_evaluation(expr['expression'], market)
                if cache_data:
                    break
        
        if cache_data and 'metrics' in cache_data:
            factors_with_metrics.append({
                'expression': expr['expression'],
                'metrics': cache_data['metrics'],
                'last_evaluated': expr['last_accessed'],
                'evaluation_count': expr['access_count'],
                'market': cache_data.get('market', 'csi300')
            })
    
    # Sort by selected metric (handle NaN values)
    factors_with_metrics.sort(
        key=lambda x: x['metrics'].get(metric, -999) if x['metrics'].get(metric) is not None and not pd.isna(x['metrics'].get(metric)) else -999, 
        reverse=True
    )
    
    # Filter out factors with invalid metrics
    valid_factors = [f for f in factors_with_metrics if f['metrics'].get(metric, -999) != -999]
    
    return jsonify({
        'success': True,
        'top_factors': valid_factors[:limit]
    })


@app.route('/api/get_recent_activity', methods=['GET'])
def get_recent_activity():
    """Get recently evaluated factors."""
    limit = request.args.get('limit', 20, type=int)
    
    # Get recent expressions ordered by last_accessed
    expressions = cache_manager.get_cached_expressions(
        limit=limit, 
        order_by='last_accessed'
    )
    
    return jsonify({
        'success': True,
        'recent_activity': expressions
    })


@app.route('/api/get_distribution_chart', methods=['POST'])
def get_distribution_chart():
    """Generate IC distribution chart data."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    result = cache_manager.get_cache(expr, market, start_date, end_date)
    
    if not result or 'daily_metrics' not in result:
        return jsonify({
            'success': False,
            'message': 'No daily data available'
        })
    
    df = pd.DataFrame(result['daily_metrics'])
    
    # Create IC distribution histogram
    fig = go.Figure()
    
    # IC histogram
    fig.add_trace(go.Histogram(
        x=df['ic'].dropna(),
        name='IC Distribution',
        nbinsx=30,
        opacity=0.7,
        marker_color='blue'
    ))
    
    fig.update_layout(
        title=f'IC Distribution - {market.upper()}',
        xaxis_title='IC Value',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'success': True,
        'chart': chart_json
    })


@app.route('/api/get_cumulative_chart', methods=['POST'])
def get_cumulative_chart():
    """Generate cumulative IC chart data."""
    data = request.json
    expr = data.get('expression', '')
    market = data.get('market', 'csi300')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    
    result = cache_manager.get_cache(expr, market, start_date, end_date)
    
    if not result or 'daily_metrics' not in result:
        return jsonify({
            'success': False,
            'message': 'No daily data available'
        })
    
    df = pd.DataFrame(result['daily_metrics'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate cumulative IC
    df['cumulative_ic'] = df['ic'].cumsum()
    df['cumulative_rank_ic'] = df['rank_ic'].cumsum()
    
    # Create chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_ic'],
        mode='lines',
        name='Cumulative IC',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_rank_ic'],
        mode='lines',
        name='Cumulative Rank IC',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Cumulative IC - {market.upper()}',
        xaxis_title='Date',
        yaxis_title='Cumulative Value',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'success': True,
        'chart': chart_json
    })


if __name__ == '__main__':
    app.run(debug=True, port=5002)