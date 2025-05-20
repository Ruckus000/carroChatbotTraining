# Multi-Phase Plan to Upgrade the NLU Benchmarking Streamlit Dashboard

This plan outlines a step-by-step approach to transform your NLU benchmarking dashboard into a more intuitive, user-friendly interface that effectively communicates complex data to both technical and non-technical users.

## Phase 1: Foundational UI Improvements

### Affected Files

- `nlu_dashboard.py` - Main dashboard file to be updated with new UI components
- `requirements-dashboard.txt` - May need updates for new dependencies
- `utils/ui_components.py` - New file to create for reusable UI components
- `assets/` - New directory to create for images and static resources
- `tests/test_ui_components.py` - New test file for UI components

### Readiness Check

- Ensure Streamlit version ≥ 1.22.0 is installed: `streamlit --version`
- Verify read access to benchmark results directory
- Confirm all required dependencies are installed: `pip install -r requirements-dashboard.txt`
- Create backup of current dashboard code: `cp nlu_dashboard.py nlu_dashboard.py.bak`

### Goal: Create a clean, intuitive layout with clear information hierarchy

1. **Create a Welcoming Homepage**

   ```python
   def render_home_page():
       st.image("assets/nlu_logo.png", width=100)
       st.title("NLU Model Performance Dashboard")

       # Key metrics overview cards in a row
       col1, col2, col3 = st.columns(3)
       with col1:
           render_metric_card("Current Model", "model_v2.3", "🤖")
       with col2:
           render_metric_card("Intent Accuracy", "94.2%", "🎯", is_percentage=True)
       with col3:
           render_metric_card("Entity F1", "87.5%", "🏷️", is_percentage=True)

       # Quick actions section
       st.subheader("Quick Actions")
       action_col1, action_col2, action_col3 = st.columns(3)
       with action_col1:
           st.button("📊 View Latest Results", on_click=set_page, args=("results",))
       with action_col2:
           st.button("📈 Performance History", on_click=set_page, args=("history",))
       with action_col3:
           st.button("❌ Error Analysis", on_click=set_page, args=("errors",))
   ```

2. **Implement a Consistent Navigation System**
   ```python
   def create_navigation():
       # Store current page in session state
       if "current_page" not in st.session_state:
           st.session_state.current_page = "home"

       # Create sidebar navigation
       with st.sidebar:
           st.image("assets/nlu_logo.png", width=80)
           st.title("Navigation")

           # Navigation buttons with visual indicators for active page
           for page, (icon, label) in PAGES.items():
               if st.session_state.current_page == page:
                   st.sidebar.button(
                       f"{icon} {label} ←",
                       key=f"nav_{page}",
                       on_click=set_page,
                       args=(page,),
                       use_container_width=True,
                   )
               else:
                   st.sidebar.button(
                       f"{icon} {label}",
                       key=f"nav_{page}",
                       on_click=set_page,
                       args=(page,),
                       use_container_width=True,
                   )
   ```

### Completion Check

- Verify sidebar navigation works across all defined pages
- Test that session state correctly maintains the current page
- Confirm metric cards render properly with all data types (string, float, etc.)
- Ensure all components are responsive and adapt to different screen sizes
- Run dashboard with `streamlit run nlu_dashboard.py` and verify UI renders correctly

## Phase 2: Enhanced Data Visualizations

### Affected Files

- `nlu_dashboard.py` - Update visualization sections
- `utils/visualization.py` - New file for visualization components
- `requirements-dashboard.txt` - Add Plotly dependency
- `tests/test_visualizations.py` - New test file for visualizations
- `utils/data_processing.py` - New/updated file for data transformation functions

### Readiness Check

- Confirm Plotly and its dependencies are correctly installed: `pip install plotly>=5.10.0`
- Verify numpy and pandas are installed and working properly
- Check that benchmark data includes required metrics for visualizations
- Test access to historical performance data for timeline charts

### Goal: Make complex data more accessible with improved visualizations

1. **Intent Performance Radar Chart**

   ```python
   def create_intent_radar_chart(metrics):
       # Extract top 5 and bottom 5 intents by F1 score
       per_class = metrics['intent_metrics']['per_class_report']
       intents = [(intent, data['f1-score']) for intent, data in per_class.items()]
       intents.sort(key=lambda x: x[1])

       bottom_5 = intents[:5]
       top_5 = intents[-5:]

       # Create radar chart data
       categories = [i[0] for i in bottom_5 + top_5]
       values = [i[1] for i in bottom_5 + top_5]

       # Radar chart using Plotly
       fig = go.Figure()

       fig.add_trace(go.Scatterpolar(
           r=values,
           theta=categories,
           fill='toself',
           name='F1 Score'
       ))

       fig.update_layout(
           polar=dict(
               radialaxis=dict(
                   visible=True,
                   range=[0, 1]
               )
           ),
           title="Best and Worst Performing Intents"
       )

       st.plotly_chart(fig, use_container_width=True)
   ```

2. **Performance History with Annotations**
   ```python
   def create_performance_timeline(history_df):
       # Create figure with secondary y-axis
       fig = make_subplots(specs=[[{"secondary_y": True}]])

       # Add traces
       fig.add_trace(
           go.Scatter(
               x=history_df['date'],
               y=history_df['intent_f1'],
               name="Intent F1",
               line=dict(color="#1f77b4", width=3)
           ),
           secondary_y=False,
       )

       fig.add_trace(
           go.Scatter(
               x=history_df['date'],
               y=history_df['entity_f1'],
               name="Entity F1",
               line=dict(color="#ff7f0e", width=3)
           ),
           secondary_y=False,
       )

       # Find significant model changes
       for i, row in history_df.iterrows():
           if i > 0 and abs(row['intent_f1'] - history_df.iloc[i-1]['intent_f1']) > 0.05:
               fig.add_annotation(
                   x=row['date'],
                   y=row['intent_f1'],
                   text="Major Change",
                   showarrow=True,
                   arrowhead=2,
               )

       # Add figure layout
       fig.update_layout(
           title_text="Performance History",
           xaxis_title="Date",
           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
           hovermode="x unified"
       )

       # Set y-axes titles
       fig.update_yaxes(title_text="F1 Score", range=[0, 1], secondary_y=False)

       st.plotly_chart(fig, use_container_width=True)
   ```

### Completion Check

- Test radar chart with different numbers of intents (few and many)
- Verify performance timeline correctly annotates significant changes
- Confirm confusion matrix heatmap renders with proper highlighting
- Check visualization responsiveness with window resizing
- Validate tooltips and hover functionality on all charts
- Run unit tests: `python -m pytest tests/test_visualizations.py`

## Phase 3: Interactive Elements and User Flows

### Affected Files

- `nlu_dashboard.py` - Add interactive components
- `utils/interactive.py` - New file for interactive components
- `utils/state_management.py` - New file for session state management
- `pages/error_explorer.py` - New file for error exploration page
- `pages/model_comparison.py` - New file for model comparison page
- `pages/guided_analysis.py` - New file for guided analysis workflow
- `tests/test_interactive_elements.py` - New test file for interactive components

### Readiness Check

- Ensure Phases 1 and 2 are fully implemented and functioning
- Verify benchmark data includes detailed results for error explorer
- Check session state functionality is working properly
- Confirm data loading functions can handle multiple model metrics simultaneously

### Goal: Add interactive elements to help users explore and understand the data

1. **Model Comparison Tool**

   ```python
   def create_model_comparison_view():
       st.header("Model Comparison")

       # Load available models
       models = load_available_models()

       # Model selection
       col1, col2 = st.columns(2)

       with col1:
           st.subheader("Select Base Model")
           base_model = st.selectbox(
               "Base model",
               options=[m["id"] for m in models],
               index=0,
               key="base_model"
           )

       with col2:
           st.subheader("Select Comparison Model")
           comparison_model = st.selectbox(
               "Comparison model",
               options=[m["id"] for m in models],
               index=min(1, len(models)-1),
               key="comparison_model"
           )

       if base_model == comparison_model:
           st.warning("Please select different models to compare")
           return

       # Load metrics for both models
       base_metrics = load_model_metrics(base_model)
       comparison_metrics = load_model_metrics(comparison_model)

       # Create comparison visualizations
       create_comparison_summary(base_metrics, comparison_metrics)
       create_detailed_comparison_tables(base_metrics, comparison_metrics)
   ```

2. **Interactive Error Explorer**
   ```python
   def create_error_explorer(metrics):
       st.header("Error Explorer")

       # Get errors from metrics
       errors = [r for r in metrics['detailed_results'] if not r['intent_correct']]

       # Create error pattern grouping
       error_patterns = {}
       for e in errors:
           pair = (e['true_intent'], e['pred_intent'])
           if pair not in error_patterns:
               error_patterns[pair] = []
           error_patterns[pair].append(e)

       # Sort patterns by frequency
       sorted_patterns = sorted(
           error_patterns.items(),
           key=lambda x: len(x[1]),
           reverse=True
       )

       # Create filter sidebar
       st.sidebar.subheader("Error Filters")

       # Filter by error pattern
       selected_pattern = st.sidebar.selectbox(
           "Error Pattern",
           options=["All Errors"] + [f"{true} → {pred} ({len(errs)})"
                                   for (true, pred), errs in sorted_patterns]
       )

       # Filter by confidence
       min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0)
       max_confidence = st.sidebar.slider("Max Confidence", 0.0, 1.0, 1.0)

       # Apply filters
       filtered_errors = errors
       if selected_pattern != "All Errors":
           pattern = selected_pattern.split(" (")[0]
           true, pred = pattern.split(" → ")
           filtered_errors = error_patterns.get((true, pred), [])

       filtered_errors = [
           e for e in filtered_errors
           if min_confidence <= e['confidence'] <= max_confidence
       ]

       # Show filtered errors
       st.write(f"Showing {len(filtered_errors)} of {len(errors)} errors")

       for i, error in enumerate(filtered_errors):
           with st.expander(f"Error {i+1}: {error['text']}"):
               col1, col2 = st.columns(2)
               col1.markdown(f"**True Intent:** {error['true_intent']}")
               col2.markdown(f"**Predicted Intent:** {error['pred_intent']}")
               st.markdown(f"**Confidence:** {error['confidence']:.4f}")

               # Show entities if available
               if 'true_entities' in error and error['true_entities']:
                   st.markdown("**Entities:**")
                   for entity in error['true_entities']:
                       st.markdown(f"- {entity['entity']}: {entity['value']}")
   ```

### Completion Check

- Test model comparison with different benchmark result files
- Verify error explorer filters work correctly with various filtering conditions
- Test guided analysis workflow from start to finish
- Confirm all interactive elements handle edge cases (no data, empty selection)
- Run integration tests: `python -m pytest tests/test_interactive_elements.py`
- Conduct user testing with sample benchmark data

## Phase 4: Polish and User Experience Enhancements

### Affected Files

- `nlu_dashboard.py` - Refine UI elements
- `utils/ui_components.py` - Update with enhanced components
- `utils/help_content.py` - New file for help text and tooltips
- `assets/css/custom.css` - Add custom CSS for styling enhancements
- `utils/export.py` - New file for export and sharing functionality
- `tests/test_ux.py` - New test file for UX components

### Readiness Check

- Confirm Phases 1-3 are completely implemented and stable
- Validate all interactive elements function as expected
- Check for any display inconsistencies across browsers
- Review user feedback from initial testing

### Goal: Refine the UI for greater usability and intuitiveness

1. **Tooltips and Contextual Help**

   ```python
   def render_metric_with_help(title, value, help_text):
       """Render a metric with a help tooltip"""
       col1, col2 = st.columns([0.9, 0.1])
       with col1:
           st.metric(label=title, value=value)
       with col2:
           with st.expander("?"):
               st.markdown(help_text)

   # Usage
   render_metric_with_help(
       "Intent F1 Score",
       f"{metrics['intent_metrics']['f1']:.2f}",
       "F1 score is the harmonic mean of precision and recall. A higher score (closer to 1.0) indicates better performance."
   )
   ```

2. **Color-Coded Performance Indicators**

   ```python
   def get_performance_color(value):
       """Returns a color based on performance value"""
       if value >= 0.9:
           return "green"
       elif value >= 0.8:
           return "lightgreen"
       elif value >= 0.7:
           return "orange"
       else:
           return "red"

   def render_performance_indicator(label, value):
       """Render a color-coded performance indicator"""
       color = get_performance_color(value)
       st.markdown(f"""
       <div style="display:flex; align-items:center; margin-bottom:10px;">
           <div style="width:120px;">{label}:</div>
           <div style="width:50px; text-align:right;">{value:.2f}</div>
           <div style="flex-grow:1; margin-left:10px;">
               <div style="width:{value*100}%; height:8px; background-color:{color}; border-radius:4px;"></div>
           </div>
       </div>
       """, unsafe_allow_html=True)

   # Usage
   render_performance_indicator("Intent F1", metrics['intent_metrics']['f1'])
   render_performance_indicator("Entity F1", entity_metrics.get('micro avg', {}).get('f1-score', 0))
   ```

### Completion Check

- Test all tooltips and help features for clarity and usefulness
- Verify color-coded performance indicators work correctly
- Confirm export functionality works for all formats
- Test email sharing feature with valid and invalid inputs
- Perform accessibility check using a tool like Wave or Axe
- Run a full system test: `python -m pytest tests/test_system.py`

## Phase 5: Documentation and Educational Elements

### Affected Files

- `nlu_dashboard.py` - Add documentation components
- `utils/tour.py` - New file for interactive tour functionality
- `utils/metrics_glossary.py` - New file for metrics definitions
- `utils/interpretation_guides.py` - New file for analysis interpretation guides
- `docs/dashboard_user_guide.md` - New user guide documentation
- `tests/test_documentation.py` - New test file for documentation components

### Readiness Check

- Ensure all previous phases are complete and tested
- Collect user feedback about unclear elements or metrics
- Confirm all components have proper tooltips and help text
- Check that the dashboard's visual design is finalized

### Goal: Help users understand how to interpret the data

1. **Interactive Tour**

   ```python
   def create_dashboard_tour():
       if "tour_step" not in st.session_state:
           st.session_state.tour_step = 0

       # Tour button in sidebar
       with st.sidebar:
           if st.button("Dashboard Tour"):
               st.session_state.tour_step = 1

       # Tour steps
       if st.session_state.tour_step > 0:
           # Create overlay with tour information
           tour_steps = [
               {"title": "Welcome to the Tour", "text": "This tour will walk you through the dashboard features.", "element": "body"},
               {"title": "Performance Summary", "text": "This section shows the key performance metrics for your model.", "element": ".performance-summary"},
               # ... more steps
           ]

           current_step = tour_steps[st.session_state.tour_step - 1]

           # Show tour dialog
           with st.sidebar:
               st.markdown(f"## {current_step['title']}")
               st.markdown(current_step['text'])

               col1, col2 = st.columns(2)
               with col1:
                   if st.button("Previous") and st.session_state.tour_step > 1:
                       st.session_state.tour_step -= 1
               with col2:
                   if st.session_state.tour_step < len(tour_steps):
                       if st.button("Next"):
                           st.session_state.tour_step += 1
                   else:
                       if st.button("Finish"):
                           st.session_state.tour_step = 0
   ```

2. **Metrics Glossary**

   ```python
   def create_metrics_glossary():
       with st.expander("Metrics Glossary"):
           st.markdown("""
           ### Metrics Explained

           #### Intent Classification Metrics
           - **Accuracy**: Percentage of examples where the predicted intent matches the true intent
           - **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
           - **Recall**: Ratio of correctly predicted positive observations to all actual positives
           - **F1 Score**: Harmonic mean of precision and recall

           #### Entity Recognition Metrics
           - **Entity Precision**: How many of the predicted entities are correct
           - **Entity Recall**: How many of the actual entities were correctly predicted
           - **Entity F1**: Harmonic mean of entity precision and recall

           #### Confusion Matrix
           A table showing how often the model confuses one intent for another
           """)
   ```

3. **Example Analysis Interpretations**
   ```python
   def add_interpretation_guidance(section):
       """Add interpretation guidance for different dashboard sections"""
       interpretations = {
           "confusion_matrix": """
           **How to interpret:**
           - Diagonal elements (top-left to bottom-right) represent correct predictions
           - Off-diagonal elements show where one intent is mistaken for another
           - Look for clusters of confusion that might indicate similar intents
           - High values off the diagonal suggest intents that might need more training data
           """,

           "error_analysis": """
           **How to interpret:**
           - Patterns in errors can reveal systemic issues in your model
           - High-confidence errors are particularly problematic as the model is "confidently wrong"
           - Similar intents appearing frequently in errors may need clearer definition
           - Consider adding more examples for intents with high error rates
           """,

           # ... more sections
       }

       if section in interpretations:
           with st.expander("Interpretation Guide"):
               st.markdown(interpretations[section])
   ```

## NLU Dashboard Navigation Guide

This comprehensive guide explains how to navigate and effectively use the redesigned NLU dashboard. It can be used both as internal documentation and as the basis for user training.

### 1. Dashboard Layout and Navigation

#### Sidebar Navigation

- **Home**: Overview of current model performance with quick actions
- **Latest Results**: Detailed analysis of the most recent benchmark run
- **Performance History**: Historical trends and version comparison
- **Error Analysis**: Interactive exploration of model errors
- **Model Comparison**: Side-by-side comparison of different model versions

**How to Navigate**:

- Use the sidebar buttons to switch between main sections
- The active page is indicated with a highlighted button and "←" symbol
- Current model info is displayed at the bottom of the sidebar

#### Common UI Elements

- **Metric Cards**: Display key performance metrics with optional trend indicators
- **Action Buttons**: Provide quick access to common functions
- **Expandable Sections**: Click on headers or arrows to expand/collapse content
- **Help Icons**: Click "?" icons or "Interpretation Guide" for contextual help

### 2. Home Page

**Purpose**: Get a quick overview of model performance and access key functions.

**Key Features**:

- **Key Metrics Overview**: View current model, intent accuracy, and entity F1 at a glance
- **Quick Actions**: Buttons for common tasks like viewing results or analyzing errors
- **Recent Performance Trend**: Chart showing performance over recent evaluations

**Navigation Tips**:

- Use Quick Action buttons to jump directly to detailed sections
- Check the performance trend to identify recent improvements or regressions
- Note any metrics highlighted in red, which may require attention

### 3. Latest Results Page

**Purpose**: Analyze detailed metrics from the most recent benchmark run.

**Key Features**:

- **Performance Metrics**: Detailed metrics for intent and entity recognition
- **Intent Performance Chart**: F1 scores for each intent, sorted by performance
- **Confusion Matrix**: Visual representation of intent classification errors
- **Export Options**: Export results in various formats (PDF, CSV, JSON, PNG)

**How to Use**:

1. Check the top metrics cards for overall performance
2. Examine the intent performance chart to identify problematic intents
3. Use the confusion matrix to identify patterns of misclassification
4. Export results as needed for sharing or documentation

### 4. Performance History Page

**Purpose**: Track model performance over time and across versions.

**Key Features**:

- **Performance Trend Chart**: Line chart showing key metrics over time
- **Model Version History**: Table of all model versions with key changes
- **Annotation of Significant Changes**: Markers for major changes in performance

**How to Use**:

1. Use the trend chart to identify long-term patterns in performance
2. Check annotations on the chart to understand significant changes
3. Refer to the version history table for details on model updates
4. Click on specific model versions for more detailed information

### 5. Error Analysis Page

**Purpose**: Identify and understand model errors to guide improvements.

**Key Features**:

- **Error Statistics**: Overview of error rates and confidence scores
- **Error Patterns Chart**: Visualization of common misclassification patterns
- **Interactive Error Explorer**: Filterable, expandable list of error examples
- **Filtering Panel**: Tools to filter errors by pattern, confidence, and entities

**How to Use**:

1. Start with the error statistics to understand the scope of errors
2. Use the error patterns chart to identify systematic issues
3. Apply filters to focus on specific error types:
   - Select an error pattern from the dropdown
   - Adjust the confidence range slider
   - Check/uncheck entity types as needed
4. Click on individual errors to see detailed information
5. Refer to the interpretation guide for help understanding the implications

**Advanced Usage**:

- Look for high-confidence errors, which indicate model confusion
- Group errors by pattern to identify similar intents that need better differentiation
- Filter by entity types to focus on entity recognition issues

### 6. Model Comparison Page

**Purpose**: Compare two model versions to evaluate improvements.

**Key Features**:

- **Model Selection**: Dropdowns to select base and comparison models
- **Metric Comparison Cards**: Side-by-side comparison of key metrics
- **Most Improved/Degraded Intents**: Tables showing biggest changes
- **Detailed Comparison Tables**: Comprehensive comparison of all metrics

**How to Use**:

1. Select a base model and comparison model from the dropdowns
2. Review the metric comparison cards to see overall changes
3. Check the tables to identify which intents improved or degraded
4. Use this information to understand the impact of model changes

### 7. Interactive Features Guide

#### Filters and Sorting

- **Error Filters**: Located in the Error Analysis sidebar
  - Pattern filter: Focus on specific error patterns
  - Confidence range: Filter by prediction confidence
  - Entity types: Show only errors with specific entities
- **Sorting Options**: Available in most tables and charts
  - Click column headers to sort tables
  - Use sort options in chart controls to reorder visualizations

#### Tooltips and Information

- Hover over data points in charts to see detailed information
- Click "?" icons or "Interpretation Guide" expanders for contextual help
- Metrics with trends show arrows indicating the direction of change

#### Export and Sharing

- Use the Export section on the Results page to download reports
- Available formats include PDF reports, CSV data, JSON data, and PNG images
- Share results via email by entering recipient addresses

### 8. Troubleshooting

**Common Issues**:

- **No data appears**: Ensure benchmark runs have been completed
- **Missing metrics**: Check that the evaluation was complete and included all metrics
- **Slow loading**: Large datasets may take time to process; refresh only when needed

**Quick Solutions**:

- Use the Refresh button on the home page to reload the latest data
- Check the console for any error messages if visualizations fail to load
- Verify that benchmark files exist in the expected directory

### 9. Best Practices

1. **Regular Benchmarking**: Run benchmarks consistently to maintain reliable history
2. **Focused Analysis**: Start with overall metrics, then drill down into specific issues
3. **Systematic Improvements**: Use error patterns to prioritize training data additions
4. **Documentation**: Export results before and after model changes to document impact
5. **Collaborative Analysis**: Share reports with team members to coordinate improvements

This guide can be accessed at any time via the "Dashboard Tour" button in the sidebar, which provides an interactive walkthrough of key features.

To align this plan with your local project:

1. **Design System Integration**

   - Replace color codes with your project's color palette
   - Update font styles and sizes to match your brand guidelines
   - Ensure component designs are consistent with your existing UI

2. **Data Structure Adaptations**

   - Modify the data loading functions to match your specific model metrics format
   - Adjust visualization components to work with your existing data schema
   - Update entity handling if your NLU system uses a different entity structure

3. **Navigation Configuration**

   - Customize the `PAGES` dictionary to include pages specific to your project:
     ```python
     PAGES = {
         "home": ("🏠", "Home"),
         "results": ("📊", "Results"),
         "history": ("📈", "History"),
         "errors": ("❌", "Error Analysis"),
         # Add your custom pages here
     }
     ```

4. **Additional Project-Specific Features**
   - Add sections for any unique aspects of your NLU system
   - Integrate with your specific model training or deployment workflows
   - Include custom metrics relevant to your use case

## Getting Started

1. Begin with Phase 1 to establish a solid foundation
2. Test with real users after each phase
3. Adjust based on feedback before moving to the next phase
4. Focus on making each component reusable to save time in later phases

This plan balances visual appeal with usability and educational elements to make your NLU benchmarking data accessible to all users, regardless of their technical background.

## Implementation Strategy Based on Code Review

After analyzing your existing codebase, here are specific strategies for implementing this plan:

### Strengths to Leverage

1. **Your Existing Styling Base**: Your custom CSS already establishes a good foundation. We can extend rather than replace it.
2. **Efficient Caching System**: Maintain your `@st.cache_data` decorators which are well implemented.
3. **Tab Organization**: Keep the tab component for sub-sections but enhance with consistent sidebar navigation.

### Priority Features to Implement First

1. **Enhanced Navigation System**: This will provide the most immediate UX improvement.
2. **Interactive Error Explorer**: Your current error analysis is good but could benefit greatly from the pattern-based filtering.
3. **Performance Timeline with Annotations**: This adds context to your existing time series plots.

### Implementation Approach

1. **Incremental Integration**: Implement features one at a time, ensuring each works before moving to the next.
2. **Maintain Backward Compatibility**: Ensure metrics files from previous runs still work with new features.
3. **Preserve Your Current Layout Strengths**: Keep your clean card-based layout but enhance with new components.
4. **Add Unit Tests**: Create tests for new features to ensure dashboard remains stable as it evolves.

By focusing on these priorities and building on your existing foundation, we can maximize impact while minimizing development time.
