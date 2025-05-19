"""
Tests for UI components in the NLU Benchmarking Dashboard
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI components
from utils.ui_components import (
    render_metric_card,
    set_page,
    create_navigation,
    render_home_page
)

class TestUIComponents(unittest.TestCase):
    """Test suite for UI components"""
    
    @patch('streamlit.container')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    def test_render_metric_card(self, mock_markdown, mock_columns, mock_container):
        """Test rendering a metric card"""
        # Setup
        mock_container.return_value.__enter__.return_value = MagicMock()
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        # Test with different inputs
        render_metric_card("Test Metric", 0.9542, "🔍")
        render_metric_card("Percentage", 95.42, is_percentage=True)
        render_metric_card("With Delta", 0.85, delta=0.05)
        
        # Assertions
        # We're mainly checking the function runs without errors
        # In a real test, we would check the specific HTML output
        mock_container.assert_called()
    
    @patch('streamlit.session_state', {})
    def test_set_page(self):
        """Test setting the current page"""
        # Mock the session state
        import streamlit as st
        st.session_state = {}
        
        # Call the function
        set_page("test_page")
        
        # Check if the page was set
        self.assertEqual(st.session_state.current_page, "test_page")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.session_state', {})
    @patch('streamlit.image')
    @patch('streamlit.title')
    def test_create_navigation(self, mock_title, mock_image, mock_session, mock_sidebar):
        """Test creating navigation sidebar"""
        # Setup mock
        mock_sidebar.button = MagicMock()
        
        # Mock the session state
        import streamlit as st
        st.session_state = {}
        
        # Create test pages dictionary
        pages = {
            "home": ("🏠", "Home"),
            "results": ("📊", "Results"),
            "history": ("📈", "History")
        }
        
        # Call the function
        create_navigation(pages)
        
        # Check if it sets default page
        self.assertEqual(st.session_state.current_page, "home")
        
        # Check if buttons were created
        # In a real test, we would check exact calls
        self.assertTrue(mock_sidebar.button.called)
    
    @patch('streamlit.columns')
    @patch('streamlit.image')
    @patch('streamlit.title')
    @patch('streamlit.subheader')
    @patch('streamlit.button')
    def test_render_home_page(self, mock_button, mock_subheader, mock_title, mock_image, mock_columns):
        """Test rendering the home page"""
        # Setup
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        # Call the function
        render_home_page()
        
        # Check if it renders the required elements
        mock_image.assert_called()
        mock_title.assert_called()
        mock_subheader.assert_called()
        
        # In a real test, we would check the specific content
        self.assertTrue(mock_button.called)

if __name__ == '__main__':
    unittest.main() 