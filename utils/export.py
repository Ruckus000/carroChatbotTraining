"""
Export utility for NLU Dashboard - Enables exporting results in various formats
"""

import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
import re
import io
import matplotlib.pyplot as plt
import plotly.io as pio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib

# Initialize Plotly with Kaleido for image export
try:
    import kaleido
    # Configure plotly to use kaleido for static image export
    pio.kaleido.scope.default_format = "png"
    pio.kaleido.scope.default_scale = 2
    pio.renderers.default = "png"
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("Warning: Kaleido package not found. Image export will be disabled.")
    print("Install kaleido with: pip install kaleido")

def get_download_link(data, filename, text):
    """
    Generate a download link for the given data
    
    Args:
        data: Data to download
        filename: Name of the file to download
        text: Text to display for the download link
    
    Returns:
        HTML string with download link
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def export_to_csv(metrics, model_id):
    """
    Export intent and entity metrics to CSV
    
    Args:
        metrics: Dictionary containing metrics
        model_id: ID of the model (for filename)
    
    Returns:
        CSV data as string
    """
    dfs = []
    
    # Intent metrics
    if 'intent_metrics' in metrics and 'per_class_report' in metrics['intent_metrics']:
        intent_data = []
        for intent, data in metrics['intent_metrics']['per_class_report'].items():
            row = {'intent': intent}
            row.update(data)
            intent_data.append(row)
        
        intent_df = pd.DataFrame(intent_data)
        intent_df = intent_df.rename(columns={
            'precision': 'intent_precision',
            'recall': 'intent_recall',
            'f1-score': 'intent_f1',
            'support': 'intent_support'
        })
        dfs.append(intent_df)
    
    # Entity metrics
    if 'entity_metrics' in metrics:
        entity_data = []
        for entity, data in metrics['entity_metrics'].items():
            if isinstance(data, dict) and 'precision' in data:
                row = {'entity': entity}
                row.update(data)
                entity_data.append(row)
        
        if entity_data:
            entity_df = pd.DataFrame(entity_data)
            entity_df = entity_df.rename(columns={
                'precision': 'entity_precision',
                'recall': 'entity_recall',
                'f1-score': 'entity_f1',
                'support': 'entity_support'
            })
            dfs.append(entity_df)
    
    # Combine and export
    if dfs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlu_metrics_{model_id}_{timestamp}.csv"
        
        # Use StringIO to create CSV data
        output = io.StringIO()
        
        # Write each dataframe with a header
        for i, df in enumerate(dfs):
            if i > 0:
                output.write("\n\n")
            if i == 0:
                output.write("INTENT METRICS\n")
            else:
                output.write("ENTITY METRICS\n")
            df.to_csv(output, index=False)
        
        return output.getvalue().encode(), filename
    
    return None, None

def export_to_json(metrics, model_id):
    """
    Export metrics to JSON
    
    Args:
        metrics: Dictionary containing metrics
        model_id: ID of the model (for filename)
    
    Returns:
        JSON data as string
    """
    if metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlu_metrics_{model_id}_{timestamp}.json"
        
        # Add export timestamp
        export_metrics = metrics.copy()
        export_metrics['export_timestamp'] = timestamp
        export_metrics['model_id'] = model_id
        
        json_data = json.dumps(export_metrics, indent=2)
        return json_data.encode(), filename
    
    return None, None

def export_plot_to_png(fig, model_id, plot_type="visualization"):
    """
    Export a Plotly figure to PNG
    
    Args:
        fig: Plotly figure to export
        model_id: ID of the model (for filename)
        plot_type: Type of plot (for filename)
    
    Returns:
        PNG image data
    """
    if fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlu_{plot_type}_{model_id}_{timestamp}.png"
        
        if not KALEIDO_AVAILABLE:
            st.error("Kaleido package is not installed. Run: pip install kaleido")
            return None, None
            
        try:
            # Convert Plotly figure to PNG
            img_bytes = pio.to_image(fig, format="png", scale=2, engine="kaleido")
            return img_bytes, filename
        except Exception as e:
            # More detailed error message
            error_message = f"Error exporting plot: {str(e)}"
            st.error(error_message)
            
            # Check if pio is using kaleido as the engine
            try:
                current_engine = pio.config.orca.executable
                if current_engine != "kaleido":
                    st.error(f"Current Plotly engine is not kaleido: {current_engine}")
            except:
                pass
            
            return None, None
    
    return None, None

def create_export_section(metrics, model_id):
    """
    Create a section for exporting results in various formats
    
    Args:
        metrics: Dictionary containing metrics
        model_id: ID of the model
    """
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    # CSV Export
    with col1:
        csv_data, csv_filename = export_to_csv(metrics, model_id)
        if csv_data:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                help="Export metrics as CSV file"
            )
        else:
            st.button("Download CSV", disabled=True, help="No data available for CSV export")
    
    # JSON Export
    with col2:
        json_data, json_filename = export_to_json(metrics, model_id)
        if json_data:
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                help="Export complete metrics as JSON file"
            )
        else:
            st.button("Download JSON", disabled=True, help="No data available for JSON export")
    
    # Email sharing
    with col3:
        if st.button("Email Results", help="Share results via email"):
            with st.expander("Email Results", expanded=True):
                recipients = st.text_input("Recipients (comma-separated emails)")
                subject = st.text_input("Subject", f"NLU Model Results: {model_id}")
                message = st.text_area("Message", f"NLU model performance results for {model_id}.\nSee attached files for details.")
                
                if st.button("Send"):
                    if not recipients:
                        st.error("Please enter at least one recipient email")
                    else:
                        # This is a placeholder - actual email sending would require SMTP configuration
                        st.success("Email would be sent (disabled in demo)")
                        st.info("In a production environment, this would send the CSV and JSON exports to the specified recipients")
                        
                        # Real implementation would use something like:
                        # send_email(recipients, subject, message, [csv_data, json_data], [csv_filename, json_filename])

def download_all_plots(plots, model_id):
    """
    Provide a download button for all plots as a zip file
    
    Args:
        plots: Dictionary of {plot_name: plotly_figure}
        model_id: ID of the model
    """
    if not plots or len(plots) == 0:
        return
        
    try:
        # Verify kaleido is available
        if not KALEIDO_AVAILABLE:
            st.error("Kaleido package is required for image export. Please install with: pip install kaleido")
            return
        
        import zipfile
        
        # Prepare a zip file in memory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlu_visualizations_{model_id}_{timestamp}.zip"
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for plot_name, fig in plots.items():
                if fig:
                    # Convert plot to PNG
                    plot_data, _ = export_plot_to_png(fig, model_id, plot_name)
                    if plot_data:
                        sanitized_name = re.sub(r'[^\w\-_.]', '_', plot_name)
                        zip_file.writestr(f"{sanitized_name}.png", plot_data)
        
        st.download_button(
            label="Download All Visualizations",
            data=zip_buffer.getvalue(),
            file_name=filename,
            mime="application/zip",
            help="Download all visualizations as a zip file"
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error creating visualization archive: {str(e)}")
        st.error(f"Detailed error: {error_details}")
        st.info("If the error mentions 'kaleido', run: pip install kaleido in your virtual environment")
        
def send_email(recipients, subject, message, attachments=None, filenames=None):
    """
    Send email with attachments (placeholder implementation)
    
    Args:
        recipients: List or comma-separated string of recipient emails
        subject: Email subject
        message: Email body
        attachments: List of attachment data (bytes)
        filenames: List of attachment filenames
    """
    # This is a placeholder that would need to be implemented with proper SMTP configuration
    st.warning("Email sending is disabled in this demo version")
    
    # Example implementation (commented out)
    """
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(',')]
        
    # Create the message
    msg = MIMEMultipart()
    msg['From'] = 'nlu-dashboard@example.com'  # Would be configured
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    
    # Add message body
    msg.attach(MIMEText(message, 'plain'))
    
    # Add attachments
    if attachments and filenames:
        for attachment, filename in zip(attachments, filenames):
            part = MIMEApplication(attachment)
            part.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(part)
    
    # Send the email
    with smtplib.SMTP('smtp.example.com', 587) as server:  # Would be configured
        server.starttls()
        server.login('username', 'password')  # Would be configured
        server.sendmail('nlu-dashboard@example.com', recipients, msg.as_string())
    """
    
    return True 