"""Tests for statistical visualization."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memento.visualization.statistical import StatisticalVisualizer

@pytest.fixture
def visualizer():
    """Create temporary visualizer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield StatisticalVisualizer(output_dir=tmpdir)

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return {
        "Method A": np.random.normal(100, 15, 50).tolist(),
        "Method B": np.random.normal(110, 15, 50).tolist(),
        "Method C": np.random.normal(105, 15, 50).tolist()
    }

@pytest.fixture
def control_treatment_data():
    """Generate control and treatment data."""
    np.random.seed(42)
    control = {
        "Dataset 1": np.random.normal(100, 15, 50).tolist(),
        "Dataset 2": np.random.normal(95, 15, 50).tolist()
    }
    treatment = {
        "Dataset 1": np.random.normal(110, 15, 50).tolist(),
        "Dataset 2": np.random.normal(105, 15, 50).tolist()
    }
    return control, treatment

def test_plot_confidence_intervals(visualizer, sample_data):
    """Test confidence interval plotting."""
    # Test with default parameters
    path = visualizer.plot_confidence_intervals(sample_data)
    assert path.exists()
    assert path.suffix == ".html"
    
    # Test with custom confidence levels
    path = visualizer.plot_confidence_intervals(
        sample_data,
        confidence_levels=[0.80, 0.90, 0.95]
    )
    assert path.exists()
    
    # Test with single dataset
    path = visualizer.plot_confidence_intervals(
        {"Single": sample_data["Method A"]}
    )
    assert path.exists()
    
    # Test with empty data
    with pytest.raises(ValueError):
        visualizer.plot_confidence_intervals({})
        
    # Test with invalid confidence levels
    with pytest.raises(ValueError):
        visualizer.plot_confidence_intervals(
            sample_data,
            confidence_levels=[0.5, 1.5]  # Invalid levels
        )

def test_plot_effect_sizes(visualizer, control_treatment_data):
    """Test effect size plotting."""
    control, treatment = control_treatment_data
    
    # Test with default parameters
    path = visualizer.plot_effect_sizes(control, treatment)
    assert path.exists()
    assert path.suffix == ".html"
    
    # Test with single dataset
    single_control = {"Dataset": control["Dataset 1"]}
    single_treatment = {"Dataset": treatment["Dataset 1"]}
    path = visualizer.plot_effect_sizes(single_control, single_treatment)
    assert path.exists()
    
    # Test with mismatched datasets
    mismatched_treatment = {
        "Dataset 1": treatment["Dataset 1"],
        "Dataset 3": treatment["Dataset 2"]  # Different key
    }
    path = visualizer.plot_effect_sizes(control, mismatched_treatment)
    assert path.exists()
    
    # Test with empty data
    with pytest.raises(ValueError):
        visualizer.plot_effect_sizes({}, {})
        
    # Test with invalid data
    with pytest.raises(ValueError):
        visualizer.plot_effect_sizes(
            {"Invalid": []},  # Empty list
            {"Invalid": [1, 2, 3]}
        )

def test_plot_power_analysis(visualizer):
    """Test power analysis plotting."""
    # Test with default parameters
    path = visualizer.plot_power_analysis(effect_size=0.5)
    assert path.exists()
    assert path.suffix == ".html"
    
    # Test with custom parameters
    path = visualizer.plot_power_analysis(
        effect_size=0.3,
        alpha=0.01,
        power_target=0.90
    )
    assert path.exists()
    
    # Test with invalid effect size
    with pytest.raises(ValueError):
        visualizer.plot_power_analysis(effect_size=0)
        
    # Test with invalid alpha
    with pytest.raises(ValueError):
        visualizer.plot_power_analysis(
            effect_size=0.5,
            alpha=1.5  # Invalid alpha
        )
        
    # Test with invalid power target
    with pytest.raises(ValueError):
        visualizer.plot_power_analysis(
            effect_size=0.5,
            power_target=1.5  # Invalid power
        )

def test_output_directory(sample_data):
    """Test output directory handling."""
    # Test with string path
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = StatisticalVisualizer(output_dir=str(tmpdir))
        path = viz.plot_confidence_intervals(sample_data)
        assert path.exists()
        
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = StatisticalVisualizer(output_dir=Path(tmpdir))
        path = viz.plot_confidence_intervals(sample_data)
        assert path.exists()
        
    # Test with non-existent directory
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_dir"
        viz = StatisticalVisualizer(output_dir=new_dir)
        path = viz.plot_confidence_intervals(sample_data)
        assert path.exists()
        
    # Test with invalid path
    with pytest.raises(ValueError):
        StatisticalVisualizer(output_dir="/invalid/path/that/cant/exist")

def test_plot_customization(visualizer, sample_data):
    """Test plot customization options."""
    # Test with custom title
    path = visualizer.plot_confidence_intervals(
        sample_data,
        title="Custom Title"
    )
    assert path.exists()
    
    # Test with custom colors
    path = visualizer.plot_confidence_intervals(
        sample_data,
        marker=dict(color="red")
    )
    assert path.exists()
    
    # Test with custom size
    path = visualizer.plot_confidence_intervals(
        sample_data,
        width=800,
        height=600
    )
    assert path.exists()
    
    # Test with custom theme
    viz = StatisticalVisualizer(
        output_dir=visualizer.output_dir,
        theme="dark"
    )
    path = viz.plot_confidence_intervals(sample_data)
    assert path.exists()

def test_error_handling(visualizer, sample_data):
    """Test error handling."""
    # Test with invalid data type
    with pytest.raises(TypeError):
        visualizer.plot_confidence_intervals([1, 2, 3])  # Not a dict
        
    # Test with invalid values
    with pytest.raises(ValueError):
        visualizer.plot_confidence_intervals(
            {"Invalid": ["not", "numbers"]}
        )
        
    # Test with NaN values
    data_with_nan = sample_data.copy()
    data_with_nan["Method A"][0] = float("nan")
    with pytest.raises(ValueError):
        visualizer.plot_confidence_intervals(data_with_nan)
        
    # Test with infinite values
    data_with_inf = sample_data.copy()
    data_with_inf["Method A"][0] = float("inf")
    with pytest.raises(ValueError):
        visualizer.plot_confidence_intervals(data_with_inf) 