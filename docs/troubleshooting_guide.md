# NLU Dashboard Troubleshooting Guide

This guide provides solutions for common issues encountered when using the NLU Benchmarking Dashboard.

## Table of Contents

1. [Installation and Startup Issues](#installation-and-startup-issues)
2. [Data Loading Issues](#data-loading-issues)
3. [Visualization Problems](#visualization-problems)
4. [Performance Issues](#performance-issues)
5. [Export and Sharing Issues](#export-and-sharing-issues)
6. [Error Messages and Solutions](#error-messages-and-solutions)

## Installation and Startup Issues

### Dashboard Won't Start

**Symptoms**:

- Error when running `run_phase4_dashboard.sh`
- Terminal error about missing modules
- Dashboard starts but immediately crashes

**Possible Causes and Solutions**:

1. **Missing Dependencies**

   - **Symptom**: Error mentions missing module (e.g., `ModuleNotFoundError: No module named 'streamlit'`)
   - **Solution**: Install the required packages
     ```bash
     pip install -r requirements-dashboard.txt
     ```

2. **Python Version Mismatch**

   - **Symptom**: Error about syntax or incompatible module version
   - **Solution**: Ensure you're using Python 3.7+
     ```bash
     python --version
     # If needed, create a virtual environment with the correct version
     python3.7 -m venv venv_dashboard
     source venv_dashboard/bin/activate
     ```

3. **Port Already in Use**

   - **Symptom**: Error message like `Address already in use: [::]:8501`
   - **Solution**: Kill the process using the port or use a different port
     ```bash
     # Find the process
     lsof -i :8501
     # Kill it
     kill -9 [PID]
     # OR run on a different port
     streamlit run nlu_dashboard.py --server.port 8502
     ```

4. **Corrupted Installation**
   - **Symptom**: Unusual errors that persist after reinstalling packages
   - **Solution**: Create a fresh virtual environment
     ```bash
     rm -rf venv_dashboard
     python3 -m venv venv_dashboard
     source venv_dashboard/bin/activate
     pip install -r requirements-dashboard.txt
     ```

## Data Loading Issues

### No Data Appears in Dashboard

**Symptoms**:

- Dashboard loads but shows "No data available"
- Empty visualizations or tables
- Error messages about missing data files

**Possible Causes and Solutions**:

1. **Missing Benchmark Results**

   - **Symptom**: Dashboard shows "No benchmark data found"
   - **Solution**:
     - Ensure benchmark results exist in the expected location
     - Check the default path in `utils/path_helpers.py`
     - Run a benchmark if none exists:
       ```bash
       python evaluate_nlu.py --output benchmark_results/latest_results.json
       ```

2. **Incorrect File Format**

   - **Symptom**: Error about JSON parsing or unexpected format
   - **Solution**:
     - Check that benchmark files are valid JSON
     - Use the correct evaluation script to generate benchmarks
     - If manually created, validate the JSON structure

3. **Permission Issues**

   - **Symptom**: File permission errors in the console
   - **Solution**:
     - Check file permissions on benchmark directories
     - Ensure the user running the dashboard has read access
     ```bash
     chmod -R 755 benchmark_results/
     ```

4. **Missing Required Metrics**
   - **Symptom**: Some visualizations show but others are empty
   - **Solution**:
     - Ensure benchmark includes all required metrics
     - Check that evaluation was complete
     - Re-run evaluation with all metrics enabled

## Visualization Problems

### Charts Not Rendering Correctly

**Symptoms**:

- Empty charts or visualizations
- Error messages in the browser console
- Partial or incorrect data shown

**Possible Causes and Solutions**:

1. **Browser Compatibility Issues**

   - **Symptom**: Visualizations work in one browser but not another
   - **Solution**:
     - Try a different modern browser (Chrome, Firefox)
     - Clear browser cache
     - Disable browser extensions that might interfere

2. **JavaScript Errors**

   - **Symptom**: Error messages in browser console
   - **Solution**:
     - Check browser console for specific errors
     - Update Streamlit and visualization libraries
     - Restart the dashboard completely

3. **Data Format Issues**

   - **Symptom**: Error about unexpected data format or missing keys
   - **Solution**:
     - Check that benchmark results have the expected structure
     - Verify that visualization code matches data format

4. **Memory or Resource Issues**
   - **Symptom**: Large visualizations fail to render
   - **Solution**:
     - Simplify complex visualizations
     - Filter data to reduce size
     - Increase browser memory limit or use a more powerful machine

### Specific Chart Troubleshooting

1. **Confusion Matrix Issues**

   - **Problem**: Matrix doesn't render or shows incorrect values
   - **Solution**:
     - Check that confusion data has correct dimensions
     - Ensure intent names are consistent
     - Try reducing the number of intents displayed

2. **Timeline Chart Issues**

   - **Problem**: Timeline data not appearing correctly
   - **Solution**:
     - Verify history data is in correct chronological format
     - Check that date formats are consistent
     - Ensure at least two data points exist for comparison

3. **Radar Chart Problems**
   - **Problem**: Radar chart distorted or empty
   - **Solution**:
     - Verify F1 scores exist for all intents
     - Check that intent names are consistent
     - Try limiting to top N intents if too many

## Performance Issues

### Dashboard Runs Slowly

**Symptoms**:

- Long loading times
- Lag when interacting with visualizations
- High CPU/memory usage

**Possible Causes and Solutions**:

1. **Large Dataset Size**

   - **Symptom**: Initial loading takes very long
   - **Solution**:
     - Filter data to only necessary metrics
     - Implement pagination for large tables
     - Optimize data loading code

2. **Too Many Visualizations**

   - **Symptom**: Page becomes sluggish with many charts
   - **Solution**:
     - Split complex pages into tabs
     - Lazy-load visualizations when needed
     - Simplify visualizations when possible

3. **Browser Resource Limitations**

   - **Symptom**: Browser becomes unresponsive
   - **Solution**:
     - Close other browser tabs and applications
     - Restart browser to free memory
     - Use a more powerful machine if available

4. **Inefficient Code**
   - **Symptom**: Specific operations take unusually long
   - **Solution**:
     - Check for inefficient loops or calculations
     - Cache results where appropriate
     - Optimize data transformation operations

## Export and Sharing Issues

### Export Functionality Not Working

**Symptoms**:

- Download button doesn't respond
- Error messages during export
- Exported files are empty or corrupted

**Possible Causes and Solutions**:

1. **Missing Permissions**

   - **Symptom**: Permission denied errors
   - **Solution**:
     - Check write permissions for the export directory
     - Run dashboard with appropriate user permissions

2. **Email Configuration Issues**

   - **Symptom**: Email sending fails
   - **Solution**:
     - Verify email configuration in settings
     - Check network/firewall settings for SMTP access
     - Use different email provider if necessary

3. **Large File Export Problems**

   - **Symptom**: Export starts but fails to complete
   - **Solution**:
     - Export smaller batches of data
     - Reduce image resolution for visualization exports
     - Check disk space availability

4. **Format Compatibility Issues**
   - **Symptom**: Export works but file can't be opened
   - **Solution**:
     - Check that exported format matches expectations
     - Try alternative export formats
     - Verify the importing application supports the format

## Error Messages and Solutions

### Common Error Messages

1. **"Failed to parse benchmark results file"**

   - **Cause**: Invalid JSON in benchmark file
   - **Solution**:
     - Check file for JSON syntax errors
     - Regenerate benchmark file
     - Verify the file isn't corrupted

2. **"No data available for selected filters"**

   - **Cause**: Filters exclude all available data
   - **Solution**:
     - Relax filter criteria
     - Check that data exists for the selected criteria
     - Reload the dashboard with default filters

3. **"Memory error during data processing"**

   - **Cause**: Operation exceeds available memory
   - **Solution**:
     - Reduce data size by filtering
     - Optimize memory-intensive operations
     - Run on a machine with more RAM

4. **"Unexpected column/key in data format"**
   - **Cause**: Data structure doesn't match expected format
   - **Solution**:
     - Check data schema consistency
     - Update code to handle dynamic data structures
     - Reformat data to match expected structure

### Platform-Specific Issues

1. **Windows-Specific Issues**

   - **Problem**: Path separators cause errors
   - **Solution**: Use os.path functions for path handling

2. **Mac-Specific Issues**

   - **Problem**: File encoding issues
   - **Solution**: Explicitly specify UTF-8 encoding when opening files

3. **Linux-Specific Issues**
   - **Problem**: Display issues with some distributions
   - **Solution**: Install additional font packages if visualizations show missing characters

## Advanced Troubleshooting

### Debugging Techniques

1. **Enable Debug Logging**

   ```bash
   streamlit run nlu_dashboard.py --logger.level=debug
   ```

2. **Check Streamlit Cache**

   - Clear the cache if you suspect caching issues:

   ```bash
   rm -rf ~/.streamlit/
   ```

3. **Inspect Network Requests**

   - Use browser developer tools to inspect network requests
   - Check for failed requests or unexpected responses

4. **Profile Performance**
   - Add timing code to identify slow operations:
   ```python
   import time
   start = time.time()
   # operation to time
   end = time.time()
   print(f"Operation took {end-start:.2f} seconds")
   ```

### Getting Help

If you've tried the solutions above and still have issues:

1. **Check Documentation**

   - Review the full dashboard documentation
   - Look for known issues in the README

2. **Ask for Help**

   - File an issue in the project repository
   - Include detailed error messages and steps to reproduce
   - Attach relevant logs and screenshots

3. **Update to Latest Version**
   - Ensure you're using the most recent dashboard version
   - Check for patches or hotfixes for known issues
