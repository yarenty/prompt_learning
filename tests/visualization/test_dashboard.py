"""Tests for real-time monitoring dashboard."""

import asyncio
from datetime import datetime, timedelta
import time

import pytest
from dash.testing.application_runners import import_app
from dash.testing.composite import DashComposite

from memento.visualization.dashboard import DashboardServer

@pytest.fixture
def dashboard():
    """Create dashboard server."""
    return DashboardServer(
        host="localhost",
        port=8050,
        update_interval=0.1,
        history_size=10
    )

@pytest.fixture
def sample_data():
    """Generate sample data."""
    return {
        "metrics": {
            "accuracy": [0.85, 0.87, 0.89],
            "latency": [0.15, 0.14, 0.13]
        },
        "resources": {
            "memory": [512, 524, 538],
            "cpu": [45, 48, 52]
        },
        "comparisons": {
            "accuracy": {
                "model1": [0.85, 0.87, 0.89],
                "model2": [0.82, 0.84, 0.86]
            }
        }
    }

def test_dashboard_initialization(dashboard):
    """Test dashboard initialization."""
    assert dashboard.host == "localhost"
    assert dashboard.port == 8050
    assert dashboard.update_interval == 0.1
    assert dashboard.history_size == 10
    
    assert isinstance(dashboard.metrics, dict)
    assert isinstance(dashboard.timestamps, list)
    assert isinstance(dashboard.resources, dict)
    assert isinstance(dashboard.comparisons, dict)

def test_update_metric(dashboard):
    """Test metric updates."""
    # Add single value
    dashboard.update_metric("accuracy", 0.85)
    assert "accuracy" in dashboard.metrics
    assert len(dashboard.metrics["accuracy"]) == 1
    assert len(dashboard.timestamps) == 1
    
    # Add multiple values
    for i in range(5):
        dashboard.update_metric("accuracy", 0.85 + i * 0.01)
        time.sleep(0.01)  # Small delay for different timestamps
        
    assert len(dashboard.metrics["accuracy"]) == 6
    assert len(dashboard.timestamps) == 6
    
    # Test history size limit
    for i in range(10):
        dashboard.update_metric("accuracy", 0.90 + i * 0.01)
        
    assert len(dashboard.metrics["accuracy"]) == dashboard.history_size
    assert len(dashboard.timestamps) == dashboard.history_size

def test_update_resource(dashboard):
    """Test resource updates."""
    # Add single value
    dashboard.update_resource("memory", 512)
    assert "memory" in dashboard.resources
    assert len(dashboard.resources["memory"]) == 1
    
    # Add multiple values
    for i in range(5):
        dashboard.update_resource("memory", 512 + i * 10)
        
    assert len(dashboard.resources["memory"]) == 6
    
    # Test history size limit
    for i in range(10):
        dashboard.update_resource("memory", 600 + i * 10)
        
    assert len(dashboard.resources["memory"]) == dashboard.history_size

def test_update_comparison(dashboard):
    """Test comparison updates."""
    # Add single value
    dashboard.update_comparison("accuracy", "model1", 0.85)
    assert "accuracy" in dashboard.comparisons
    assert "model1" in dashboard.comparisons["accuracy"]
    assert len(dashboard.comparisons["accuracy"]["model1"]) == 1
    
    # Add multiple values
    for i in range(5):
        dashboard.update_comparison("accuracy", "model1", 0.85 + i * 0.01)
        
    assert len(dashboard.comparisons["accuracy"]["model1"]) == 6
    
    # Add second model
    dashboard.update_comparison("accuracy", "model2", 0.82)
    assert "model2" in dashboard.comparisons["accuracy"]
    
    # Test history size limit
    for i in range(10):
        dashboard.update_comparison("accuracy", "model1", 0.90 + i * 0.01)
        
    assert len(dashboard.comparisons["accuracy"]["model1"]) == dashboard.history_size

def test_clear_data(dashboard, sample_data):
    """Test data clearing."""
    # Add sample data
    for metric, values in sample_data["metrics"].items():
        for value in values:
            dashboard.update_metric(metric, value)
            
    for resource, values in sample_data["resources"].items():
        for value in values:
            dashboard.update_resource(resource, value)
            
    for metric, models in sample_data["comparisons"].items():
        for model, values in models.items():
            for value in values:
                dashboard.update_comparison(metric, model, value)
                
    # Clear data
    dashboard.clear_data()
    
    assert len(dashboard.metrics) == 0
    assert len(dashboard.timestamps) == 0
    assert len(dashboard.resources) == 0
    assert len(dashboard.comparisons) == 0

@pytest.mark.asyncio
async def test_dashboard_server(dashboard):
    """Test dashboard server start/stop."""
    # Start server in background
    server_task = asyncio.create_task(dashboard.start())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    # Add some data
    dashboard.update_metric("accuracy", 0.85)
    dashboard.update_resource("memory", 512)
    dashboard.update_comparison("accuracy", "model1", 0.85)
    
    # Stop server
    await dashboard.stop()
    
    # Cancel server task
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

@pytest.mark.integration
def test_dashboard_integration(dashboard, sample_data):
    """Test dashboard integration with sample data."""
    # Add sample data
    for metric, values in sample_data["metrics"].items():
        for value in values:
            dashboard.update_metric(metric, value)
            time.sleep(0.01)  # Small delay for different timestamps
            
    for resource, values in sample_data["resources"].items():
        for value in values:
            dashboard.update_resource(resource, value)
            
    for metric, models in sample_data["comparisons"].items():
        for model, values in models.items():
            for value in values:
                dashboard.update_comparison(metric, model, value)
                
    # Check data integrity
    for metric in sample_data["metrics"]:
        assert metric in dashboard.metrics
        assert len(dashboard.metrics[metric]) == len(sample_data["metrics"][metric])
        
    for resource in sample_data["resources"]:
        assert resource in dashboard.resources
        assert len(dashboard.resources[resource]) == len(sample_data["resources"][resource])
        
    for metric, models in sample_data["comparisons"].items():
        assert metric in dashboard.comparisons
        for model in models:
            assert model in dashboard.comparisons[metric]
            assert len(dashboard.comparisons[metric][model]) == len(models[model])

@pytest.mark.stress
def test_dashboard_stress(dashboard):
    """Test dashboard under stress."""
    # Add many metrics rapidly
    for i in range(1000):
        dashboard.update_metric(f"metric_{i}", float(i))
        dashboard.update_resource(f"resource_{i}", float(i))
        dashboard.update_comparison(f"metric_{i}", f"model_{i}", float(i))
        
    # Check history size limits
    for metric in dashboard.metrics:
        assert len(dashboard.metrics[metric]) <= dashboard.history_size
        
    for resource in dashboard.resources:
        assert len(dashboard.resources[resource]) <= dashboard.history_size
        
    for metric in dashboard.comparisons:
        for model in dashboard.comparisons[metric]:
            assert len(dashboard.comparisons[metric][model]) <= dashboard.history_size
            
    # Check timestamp consistency
    assert len(dashboard.timestamps) <= dashboard.history_size
    for i in range(1, len(dashboard.timestamps)):
        assert dashboard.timestamps[i] > dashboard.timestamps[i-1] 