# NLU GUI Launcher Implementation Plan with SOLID & DRY Principles

**Assumptions:**

- Your project root directory is where `gui_launcher.py` will be created.
- `api.py` is located at `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py`.
- This implementation is specifically optimized for macOS.

---

**Project Goal:** Create a `gui_launcher.py` that uses NiceGUI to:

1.  Start/Stop the `api.py` server.
2.  Display the server's status.
3.  Capture and display all messages sent to and received by the `/api/dialog` endpoint.
4.  Capture and display any errors or general output from the `api.py` server process.

**Architecture Overview:**
Following SOLID principles, we'll structure our application with these components:

- `ApiProcessManager`: Manages the API process lifecycle (Single Responsibility)
- `LogHandler`: Processes and filters log streams (Single Responsibility)
- `UiComponents`: Manages UI element creation and updates (Single Responsibility)
- `AppConfig`: Handles application configuration (Single Responsibility)
- Main application class to orchestrate components (Dependency Injection)

---

## Phase 0: Setup and Basic NiceGUI Structure

**Actions:**

- [ ] **Install NiceGUI:**
  - [ ] Command: `pip install nicegui`
- [ ] **Create Project Structure:**
  - [ ] Create directory: `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui`
  - [ ] Create files:
    - [ ] `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py` (main entry point)
    - [ ] `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/api_manager.py` (API process management)
    - [ ] `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/log_handler.py` (Log processing)
    - [ ] `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/ui_components.py` (UI components)
    - [ ] `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/config.py` (Configuration)
- [ ] **Create Basic Config Module:**

  - [ ] Action: Create `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/config.py` with the following content:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/config.py
    import os

    class AppConfig:
        """Configuration for the GUI launcher application"""
        def __init__(self, api_script_path=None):
            # Allow custom path or use default relative path
            self.api_script_path = api_script_path or os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "api.py"
            )
            self.app_title = "NLU API Control"
            self.max_log_lines = 200

            # API configuration settings
            self.api_port = 8001  # Default port
            self.api_port_method = "auto"  # How port is passed to API: "auto", "arg", "env", "none"

            # Auto-detect API port configuration method
            self._detect_api_port_method()

        def _detect_api_port_method(self):
            """
            Detect how the API script accepts port configuration.
            This is important for different API frameworks:
            - FastAPI/Uvicorn often use --port or PORT env var
            - Flask might use --port or FLASK_PORT env var
            - Django uses different approaches
            """
            if not os.path.exists(self.api_script_path):
                return

            # Read the API script to detect how it might handle port config
            try:
                with open(self.api_script_path, 'r') as f:
                    content = f.read().lower()

                    # Check for different port configuration patterns
                    if 'uvicorn.run' in content and 'port' in content:
                        self.api_port_method = "arg"  # Likely FastAPI using --port
                    elif 'fastapi' in content:
                        self.api_port_method = "arg"  # FastAPI typically uses --port
                    elif 'flask' in content and 'port=' in content:
                        self.api_port_method = "arg"  # Flask app.run(port=...)
                    elif 'os.environ.get(' in content and ('port' in content.lower() or 'PORT' in content):
                        self.api_port_method = "env"  # Environment variable
                    elif 'argparse' in content and 'port' in content:
                        self.api_port_method = "arg"  # Command line args
                    else:
                        # Could not definitively determine, use fallback
                        self.api_port_method = "arg"  # Most common approach

            except (IOError, UnicodeDecodeError):
                # File read error, use default
                self.api_port_method = "arg"

        def validate(self):
            """Validate the configuration"""
            if not os.path.exists(self.api_script_path):
                return False, f"API script not found at {self.api_script_path}"

            # Validate port range
            if not (1024 <= self.api_port <= 65535):
                return False, f"Invalid port number: {self.api_port} (must be between 1024 and 65535)"

            return True, "Configuration valid"

        def get_api_command(self):
            """
            Get command to start API with proper configuration options
            based on the detected port method
            """
            cmd = ['python', self.api_script_path]

            # Add port configuration based on detected method
            if self.api_port_method == "arg":
                # Try common command line argument patterns
                cmd.extend(['--port', str(self.api_port)])
            elif self.api_port_method == "env":
                # Will be handled by environment variables
                pass  # Environment vars are set elsewhere

            return cmd

        def get_api_env(self):
            """Get environment variables for API process"""
            env = os.environ.copy()  # Start with current environment

            # Add port configuration if needed
            if self.api_port_method == "env":
                # Set common environment variables for port configuration
                env['PORT'] = str(self.api_port)
                env['API_PORT'] = str(self.api_port)
                env['FLASK_RUN_PORT'] = str(self.api_port)

            return env
    ```

- [ ] **Create Main Entry Point:**

  - [ ] Action: Create `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py` with the following content:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py
    from nicegui import ui, app
    from gui.config import AppConfig
    from gui.api_manager import ApiProcessManager
    from gui.log_handler import LogHandler

    class NluGuiLauncher:
        """Main application class that orchestrates the components"""
        def __init__(self):
            # Initialize configuration
            self.config = AppConfig()

            # UI state variables
            self.status_label = None
            self.start_button = None
            self.stop_button = None
            self.messages_log_area = None
            self.server_log_area = None

            # Initialize components
            self.log_handler = LogHandler(
                on_api_message=self._on_api_message,
                on_server_log=self._on_server_log
            )

            self.api_manager = ApiProcessManager(
                api_script_path=self.config.api_script_path,
                on_output_callback=self.log_handler.process_log_line,
                on_status_change=self._on_status_change
            )

            # Register shutdown handler
            app.on_shutdown(self._on_shutdown)

        def _on_api_message(self, message):
            """Handle API message (request/response)"""
            if self.messages_log_area:
                self.messages_log_area.push(message)

        def _on_server_log(self, message, is_error=False):
            """Handle server log message"""
            if self.server_log_area:
                # If NiceGUI logs support HTML or special formatting for errors
                # you can implement it here
                if is_error:
                    # Handle error formatting if supported
                    # For now, just push the message
                    self.server_log_area.push(message)
                else:
                    self.server_log_area.push(message)

        def _on_status_change(self, status, is_running):
            """Handle API status change"""
            if self.status_label:
                self.status_label.set_text(f"API Status: {status}")

            if self.start_button:
                self.start_button.enable() if not is_running else self.start_button.disable()

            if self.stop_button:
                self.stop_button.enable() if is_running else self.stop_button.disable()

        def _start_api(self):
            """Start the API server with configured parameters"""
            # Get command and environment based on configuration
            command = self.config.get_api_command()
            env = self.config.get_api_env()

            # Start the API process with the configured settings
            success, message = self.api_manager.start(command=command, env=env)
            ui.notify(message, type='success' if success else 'negative')

        def _stop_api(self):
            """Stop the API server"""
            success, message = self.api_manager.stop()
            ui.notify(message, type='success' if success else 'negative')

        async def _on_shutdown(self):
            """Handle application shutdown"""
            print("GUI shutting down, ensuring API server is stopped.")
            if self.api_manager.is_running():
                self.api_manager.stop()

        def _clear_messages_log(self):
            """Clear messages log area"""
            if self.messages_log_area:
                self.messages_log_area.clear()

        def _clear_server_log(self):
            """Clear server log area"""
            if self.server_log_area:
                self.server_log_area.clear()

        def setup_ui(self):
            """Set up the UI components"""
            # Main card container
            with ui.card().tight().classes('w-full max-w-2xl mx-auto'):
                ui.label("NLU API Control Panel").classes('text-h6 p-4 bg-primary text-white text-center')

                # Status and control section
                with ui.card_section():
                    self.status_label = ui.label("API Status: Stopped").classes('text-center mb-2')
                    with ui.row().classes('w-full justify-around mb-4'):
                        self.start_button = ui.button("Start API Server", on_click=self._start_api).props('color=positive')
                        self.stop_button = ui.button("Stop API Server", on_click=self._stop_api).props('color=negative')
                        self.stop_button.disable()  # Initially disabled

                ui.separator()

                # Log areas
                with ui.card_section().classes('w-full'):
                    ui.label("API Messages Log (Requests/Responses)").classes('text-subtitle1 mb-1')
                    self.messages_log_area = ui.log(max_lines=self.config.max_log_lines).classes('w-full h-40 font-mono text-xs border p-1')

                with ui.card_section().classes('w-full mt-4'):
                    ui.label("Raw Server Output / Errors").classes('text-subtitle1 mb-1')
                    self.server_log_area = ui.log(max_lines=self.config.max_log_lines).classes('w-full h-40 font-mono text-xs border p-1')

                # Control buttons for logs
                with ui.row().classes('w-full justify-around mt-2 pb-2'):
                    ui.button("Clear Messages Log", on_click=self._clear_messages_log).props('outline size=sm')
                    ui.button("Clear Server Log", on_click=self._clear_server_log).props('outline size=sm')

                # Add configuration section for API port
                with ui.card_section():
                    with ui.expansion("Configuration", icon="settings").classes('w-full'):
                        with ui.row().classes('items-center'):
                            ui.label("API Port:").classes('mr-2')
                            port_input = ui.number(value=self.config.api_port, min=1024, max=65535)
                            ui.button("Apply", on_click=lambda: self._on_port_change(int(port_input.value))).props('outline size=sm')

        def _on_port_change(self, new_port):
            """Handle port configuration change"""
            if self.api_manager.is_running():
                ui.notify("Stop the API server before changing port", type='warning')
                return

            self.config.api_port = new_port
            ui.notify(f"Port updated to {new_port}", type='positive')

    def main():
        # Initialize and run the application
        launcher = NluGuiLauncher()
        launcher.setup_ui()
        ui.run(title=launcher.config.app_title)

    if __name__ in {"__main__", "__mp_main__"}:
        main()
    ```

**Objective Check (Phase 0):**

- [ ] **Cursor Action:** Execute the following:
  - [ ] Verify the directory structure is created correctly.
  - [ ] Verify that all files exist with the correct content.
  - [ ] Run the command: `python /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py`
- [ ] **Expected Outcome:**
  - [ ] A browser window opens automatically.
  - [ ] The browser window displays the text "NLU API Control Panel".
  - [ ] No errors are printed in the console where `gui_launcher.py` was executed.
- [ ] **Confirmation:** State "Phase 0 completed successfully. All objectives met." if all checkboxes are true. Otherwise, STOP and report issues.

---

## Phase 1: API Process Manager Implementation

**Actions:**

- [ ] **Create API Process Manager:**

  - [ ] Action: Create `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/api_manager.py` with the following content:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/api_manager.py
    import os
    import signal
    import subprocess
    import threading
    from typing import Callable, Optional, List, Tuple

    class ApiProcessManager:
        """
        Manages the lifecycle of the API process.
        Follows Single Responsibility Principle by focusing only on process management.
        Optimized specifically for macOS.
        """
        def __init__(self, api_script_path: str,
                     on_output_callback: Optional[Callable[[str, str], None]] = None,
                     on_status_change: Optional[Callable[[str, bool], None]] = None):
            self.api_script_path = api_script_path
            self.process = None
            self.on_output_callback = on_output_callback
            self.on_status_change = on_status_change
            self._output_threads = []

        def start(self, command=None, env=None) -> Tuple[bool, str]:
            """
            Start the API process.
            Returns: (success, message)
            """
            # Check if already running
            if self.is_running():
                return False, "API server is already running"

            # Check if script exists
            if not os.path.exists(self.api_script_path):
                return False, f"API script not found at {self.api_script_path}"

            try:
                # Start the process with macOS-specific settings
                cmd = command or ['python', self.api_script_path]

                # Use provided environment or system environment
                process_env = env or os.environ.copy()

                # Create process with macOS-specific settings
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,  # Create new process group on macOS
                    cwd=os.path.dirname(self.api_script_path),
                    env=process_env
                )

                # Start output reader threads
                self._start_output_readers()

                # Notify status change
                if self.on_status_change:
                    self.on_status_change(f"Running (PID: {self.process.pid})", True)

                return True, f"API server started with PID: {self.process.pid}"

            except FileNotFoundError:
                return False, f"Python interpreter not found or API script not found"
            except PermissionError:
                return False, f"Permission denied when trying to run API script"
            except Exception as e:
                return False, f"Error starting API server: {e}"

        def stop(self) -> Tuple[bool, str]:
            """
            Stop the API process with macOS-specific handling.
            Returns: (success, message)
            """
            if not self.is_running():
                return False, "API server is not running"

            try:
                # macOS process termination with multiple approaches
                # First try SIGTERM directly to the process
                os.kill(self.process.pid, signal.SIGTERM)

                # Also try to terminate the process group (important for macOS)
                try:
                    pgid = os.getpgid(self.process.pid)
                    if pgid > 0:  # Valid process group
                        os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Process might already be terminating, continue
                    pass

                # Wait for process to terminate
                try:
                    self.process.wait(timeout=5)
                    success = True
                    message = "API server stopped"
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination times out
                    # Try process group first
                    try:
                        pgid = os.getpgid(self.process.pid)
                        if pgid > 0:
                            os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        # Try direct kill instead
                        self.process.kill()

                    success = True
                    message = "API server stopped (force killed)"

                # Notify status change
                if self.on_status_change:
                    self.on_status_change("Stopped", False)

                # Cleanup
                self.process = None
                return success, message

            except Exception as e:
                # Try force kill as last resort
                try:
                    if self.process and self.process.poll() is None:
                        self.process.kill()
                        if self.on_status_change:
                            self.on_status_change("Stopped (Force killed)", False)
                        return True, "API server force killed after error"
                except:
                    pass
                return False, f"Error stopping API server: {e}"

        def is_running(self) -> bool:
            """Check if the API process is running"""
            return self.process is not None and self.process.poll() is None

        def _start_output_readers(self):
            """Start threads to read stdout and stderr from the process"""
            if not self.process:
                return

            # Clear previous threads
            self._output_threads = []

            # Start new reader threads
            stdout_thread = threading.Thread(
                target=self._read_output_stream,
                args=(self.process.stdout, 'stdout'),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._read_output_stream,
                args=(self.process.stderr, 'stderr'),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            self._output_threads = [stdout_thread, stderr_thread]

        def _read_output_stream(self, stream, stream_name):
            """Read lines from a stream and pass to callback"""
            try:
                for line in iter(stream.readline, ''):
                    line = line.strip()
                    if not line:
                        continue

                    # Call the callback with the line and stream name
                    if self.on_output_callback:
                        self.on_output_callback(line, stream_name)

                    # Check if process has ended - might need to update status
                    if not self.is_running() and self.process:
                        if self.on_status_change:
                            self.on_status_change("Stopped (Process ended)", False)
                        self.process = None
                        break

                stream.close()
            except Exception as e:
                print(f"Error reading from {stream_name}: {e}")
    ```

- [ ] **Create Log Handler:**

  - [ ] Action: Create `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/log_handler.py` with the following content:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/log_handler.py
    import re
    from typing import Optional, Callable, Dict, List

    class LogHandler:
        """
        Handles and filters log messages.
        Follows Single Responsibility Principle for log processing.
        """
        def __init__(self,
                     on_api_message: Optional[Callable[[str], None]] = None,
                     on_server_log: Optional[Callable[[str, bool], None]] = None,
                     api_patterns: Optional[List[str]] = None):
            self.on_api_message = on_api_message
            self.on_server_log = on_server_log

            # Configurable log patterns
            default_patterns = [r'API_REQUEST:', r'API_RESPONSE:', r'API_ERROR:']
            self.api_patterns = api_patterns or default_patterns

            # Compile regex patterns for better performance
            self.api_msg_pattern = re.compile('|'.join(self.api_patterns))

            # Different logger prefix patterns based on common formats
            self.logger_prefix_patterns = [
                # Standard Python logging format: "INFO:module:message"
                re.compile(r'^[A-Z]+:\S+?:(.*)'),
                # Timestamp format: "[2023-05-10 12:34:56] INFO message"
                re.compile(r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s+[A-Z]+\s+(.*)'),
                # Common format: "2023-05-10 12:34:56,789 - module - INFO - message"
                re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?\s+-\s+\S+\s+-\s+[A-Z]+\s+-\s+(.*)'),
                # Uvicorn/Starlette format: "INFO: message"
                re.compile(r'^[A-Z]+:\s+(.*)'),
            ]

            # Error pattern detection
            self.error_patterns = [
                'ERROR', 'WARN', 'CRITICAL', 'EXCEPTION', 'FAIL', 'FAILED'
            ]

        def clean_log_prefix(self, line: str) -> str:
            """Try to clean log prefix using various patterns"""
            for pattern in self.logger_prefix_patterns:
                match = pattern.match(line)
                if match:
                    return match.group(1).strip()
            return line

        def is_error_message(self, line: str, stream_name: str) -> bool:
            """Check if a log line represents an error message"""
            if stream_name == 'stderr':
                return True

            # Check for error keywords
            upper_line = line.upper()
            return any(error_pattern in upper_line for error_pattern in self.error_patterns)

        def process_log_line(self, line: str, stream_name: str):
            """Process a line of log output and route to appropriate handlers"""
            # Print to console for debugging
            print(f"FROM_API [{stream_name.upper()}]: {line}")

            # Check if this is an API message (request/response)
            is_api_msg = self.api_msg_pattern.search(line) is not None

            if is_api_msg and self.on_api_message:
                # Clean up logger prefix if present
                cleaned_line = self.clean_log_prefix(line)
                self.on_api_message(cleaned_line)

            # Send to server log if it's not an API message or if it's from stderr
            # (we want errors to appear in both logs if they're API errors)
            if self.on_server_log:
                # Add prefix for error streams or error-like messages
                is_error = self.is_error_message(line, stream_name)
                prefix = f"[{stream_name.upper()}] " if is_error else ""

                # Pass both the message and error status
                self.on_server_log(f"{prefix}{line}", is_error)
    ```

- [ ] **Update Main Entry Point:**

  - [ ] Action: Update `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py` to use the new modules:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py
    from nicegui import ui, app
    from gui.config import AppConfig
    from gui.api_manager import ApiProcessManager
    from gui.log_handler import LogHandler

    class NluGuiLauncher:
        """Main application class that orchestrates the components"""
        def __init__(self):
            # Initialize configuration
            self.config = AppConfig()

            # UI state variables
            self.status_label = None
            self.start_button = None
            self.stop_button = None
            self.messages_log_area = None
            self.server_log_area = None

            # Initialize components
            self.log_handler = LogHandler(
                on_api_message=self._on_api_message,
                on_server_log=self._on_server_log
            )

            self.api_manager = ApiProcessManager(
                api_script_path=self.config.api_script_path,
                on_output_callback=self.log_handler.process_log_line,
                on_status_change=self._on_status_change
            )

            # Register shutdown handler
            app.on_shutdown(self._on_shutdown)

        def _on_api_message(self, message):
            """Handle API message (request/response)"""
            if self.messages_log_area:
                self.messages_log_area.push(message)

        def _on_server_log(self, message, is_error=False):
            """Handle server log message"""
            if self.server_log_area:
                # If NiceGUI logs support HTML or special formatting for errors
                # you can implement it here
                if is_error:
                    # Handle error formatting if supported
                    # For now, just push the message
                    self.server_log_area.push(message)
                else:
                    self.server_log_area.push(message)

        def _on_status_change(self, status, is_running):
            """Handle API status change"""
            if self.status_label:
                self.status_label.set_text(f"API Status: {status}")

            if self.start_button:
                self.start_button.enable() if not is_running else self.start_button.disable()

            if self.stop_button:
                self.stop_button.enable() if is_running else self.stop_button.disable()

        def _start_api(self):
            """Start the API server with configured parameters"""
            # Get command and environment based on configuration
            command = self.config.get_api_command()
            env = self.config.get_api_env()

            # Start the API process with the configured settings
            success, message = self.api_manager.start(command=command, env=env)
            ui.notify(message, type='success' if success else 'negative')

        def _stop_api(self):
            """Stop the API server"""
            success, message = self.api_manager.stop()
            ui.notify(message, type='success' if success else 'negative')

        async def _on_shutdown(self):
            """Handle application shutdown"""
            print("GUI shutting down, ensuring API server is stopped.")
            if self.api_manager.is_running():
                self.api_manager.stop()

        def _clear_messages_log(self):
            """Clear messages log area"""
            if self.messages_log_area:
                self.messages_log_area.clear()

        def _clear_server_log(self):
            """Clear server log area"""
            if self.server_log_area:
                self.server_log_area.clear()

        def setup_ui(self):
            """Set up the UI components"""
            # Main card container
            with ui.card().tight().classes('w-full max-w-2xl mx-auto'):
                ui.label("NLU API Control Panel").classes('text-h6 p-4 bg-primary text-white text-center')

                # Status and control section
                with ui.card_section():
                    self.status_label = ui.label("API Status: Stopped").classes('text-center mb-2')
                    with ui.row().classes('w-full justify-around mb-4'):
                        self.start_button = ui.button("Start API Server", on_click=self._start_api).props('color=positive')
                        self.stop_button = ui.button("Stop API Server", on_click=self._stop_api).props('color=negative')
                        self.stop_button.disable()  # Initially disabled

                ui.separator()

                # Log areas
                with ui.card_section().classes('w-full'):
                    ui.label("API Messages Log (Requests/Responses)").classes('text-subtitle1 mb-1')
                    self.messages_log_area = ui.log(max_lines=self.config.max_log_lines).classes('w-full h-40 font-mono text-xs border p-1')

                with ui.card_section().classes('w-full mt-4'):
                    ui.label("Raw Server Output / Errors").classes('text-subtitle1 mb-1')
                    self.server_log_area = ui.log(max_lines=self.config.max_log_lines).classes('w-full h-40 font-mono text-xs border p-1')

                # Control buttons for logs
                with ui.row().classes('w-full justify-around mt-2 pb-2'):
                    ui.button("Clear Messages Log", on_click=self._clear_messages_log).props('outline size=sm')
                    ui.button("Clear Server Log", on_click=self._clear_server_log).props('outline size=sm')

                # Add configuration section for API port
                with ui.card_section():
                    with ui.expansion("Configuration", icon="settings").classes('w-full'):
                        with ui.row().classes('items-center'):
                            ui.label("API Port:").classes('mr-2')
                            port_input = ui.number(value=self.config.api_port, min=1024, max=65535)
                            ui.button("Apply", on_click=lambda: self._on_port_change(int(port_input.value))).props('outline size=sm')

        def _on_port_change(self, new_port):
            """Handle port configuration change"""
            if self.api_manager.is_running():
                ui.notify("Stop the API server before changing port", type='warning')
                return

            self.config.api_port = new_port
            ui.notify(f"Port updated to {new_port}", type='positive')

    def main():
        # Initialize and run the application
        launcher = NluGuiLauncher()
        launcher.setup_ui()
        ui.run(title=launcher.config.app_title)

    if __name__ in {"__main__", "__mp_main__"}:
        main()
    ```

**Objective Check (Phase 1):**

- [ ] **Cursor Action:**
  - [ ] Verify all files have been created/updated with the correct content.
  - [ ] Run `python /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py`.
  - [ ] In the opened browser window, click the "Start API Server" button.
  - [ ] After a few seconds, check if the API is responding (e.g., open a new terminal and run `curl http://localhost:8001/api/health` or `curl http://localhost:8000/api/health` depending on what port your `api.py` uses. Adjust the port in the curl command if your `api.py` uses a different one, common is 8000 or 8001).
  - [ ] In the GUI, click the "Stop API Server" button.
  - [ ] After a few seconds, try the `curl` command again to verify the API is no longer responding.
- [ ] **Expected Outcome:**
  - [ ] GUI launches without console errors.
  - [ ] "Start API Server" button click:
    - [ ] Status label updates to "API Status: Running (PID: ...)".
    - [ ] "Start" button becomes disabled, "Stop" button becomes enabled.
    - [ ] `curl` command to `/api/health` (or similar) receives a successful response (e.g., `{"status": "healthy"}`).
    - [ ] A python process running `api.py` is visible in the system's process/task manager.
  - [ ] "Stop API Server" button click:
    - [ ] Status label updates to "API Status: Stopped".
    - [ ] "Stop" button becomes disabled, "Start" button becomes enabled.
    - [ ] `curl` command to `/api/health` fails (e.g., "Connection refused").
    - [ ] The python process running `api.py` is terminated.
  - [ ] Closing the GUI browser window/tab also stops the API server process if it was running (check console output of `gui_launcher.py` for "GUI shutting down..." message and verify process termination).
- [ ] **Confirmation:** State "Phase 1 completed successfully. All objectives met." if all checkboxes are true. Otherwise, STOP and report issues.

---

## Phase 2: UI Components Extraction (Applying Open/Closed Principle)

**Actions:**

- [ ] **Create UI Components Module:**

  - [ ] Action: Create `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/ui_components.py` with the following content:
  - [ ] Code:

    ```python
    # /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/ui_components.py
    from nicegui import ui
    from typing import Callable, Optional

    class UiComponents:
        """
        UI Components for the NLU GUI Launcher.
        Following Open/Closed Principle, this module is open for extension
        """
        @staticmethod
        def create_header(title: str):
            """Create application header with macOS styling"""
            return ui.label(title).classes('text-h6 p-4 bg-primary text-white text-center rounded-t-lg')

        @staticmethod
        def create_status_section(status_text: str, on_start: Callable, on_stop: Callable):
            """Create status section with control buttons"""
            status_label = ui.label(f"API Status: {status_text}").classes('text-center mb-2')

            with ui.row().classes('w-full justify-around mb-4'):
                start_button = ui.button("Start API Server", on_click=on_start).props('color=positive no-caps')
                stop_button = ui.button("Stop API Server", on_click=on_stop).props('color=negative no-caps')

                # Initially stop button is disabled
                if status_text == "Stopped" or status_text.startswith("Stopped"):
                    stop_button.disable()

            return status_label, start_button, stop_button

        @staticmethod
        def create_log_section(title: str, max_lines: int = 200):
            """Create a log section with title and log area"""
            ui.label(title).classes('text-subtitle1 mb-1')
            log_area = ui.log(max_lines=max_lines).classes('w-full h-40 font-mono text-xs border p-1 rounded')
            return log_area

        @staticmethod
        def create_log_controls(on_clear_messages: Callable, on_clear_server: Callable):
            """Create log control buttons"""
            with ui.row().classes('w-full justify-around mt-2 pb-2'):
                ui.button("Clear Messages Log", on_click=on_clear_messages).props('outline size=sm no-caps')
                ui.button("Clear Server Log", on_click=on_clear_server).props('outline size=sm no-caps')

        @staticmethod
        def create_config_section(current_port: int, on_port_change: Callable[[int], None]):
            """Create configuration section with macOS-friendly styling"""
            with ui.expansion("Configuration", icon="settings").classes('w-full'):
                with ui.row().classes('items-center'):
                    ui.label("API Port:").classes('mr-2')
                    port_input = ui.number(value=current_port, min=1024, max=65535)
                    ui.button("Apply", on_click=lambda: on_port_change(int(port_input.value))).props('outline size=sm no-caps')
            return port_input
    ```

- [ ] **Update Main Application to Use UI Components:**

  - [ ] Action: Update `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py` to use the UI components module:
  - [ ] Code (only showing the modified `setup_ui` method, rest remains the same):

    ```python
    # Partial update to /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py - only the setup_ui method
    from gui.ui_components import UiComponents

    # ... [rest of the class remains the same]

    def setup_ui(self):
        """Set up the UI components using the UiComponents module"""
        # Main card container
        with ui.card().tight().classes('w-full max-w-2xl mx-auto'):
            # Create header
            UiComponents.create_header("NLU API Control Panel")

            # Status and control section
            with ui.card_section():
                self.status_label, self.start_button, self.stop_button = UiComponents.create_status_section(
                    status_text="Stopped",
                    on_start=self._start_api,
                    on_stop=self._stop_api
                )

            ui.separator()

            # Log areas
            with ui.card_section().classes('w-full'):
                self.messages_log_area = UiComponents.create_log_section(
                    "API Messages Log (Requests/Responses)",
                    max_lines=self.config.max_log_lines
                )

            with ui.card_section().classes('w-full mt-4'):
                self.server_log_area = UiComponents.create_log_section(
                    "Raw Server Output / Errors",
                    max_lines=self.config.max_log_lines
                )

            # Control buttons for logs
            UiComponents.create_log_controls(
                on_clear_messages=self._clear_messages_log,
                on_clear_server=self._clear_server_log
            )

    # ... [rest of the file remains the same]
    ```

**Objective Check (Phase 2):**

- [ ] **Cursor Action:**
  - [ ] Verify the `ui_components.py` file has been created with the correct content.
  - [ ] Verify the `setup_ui` method in `gui_launcher.py` has been updated correctly.
  - [ ] Run `python /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py`.
  - [ ] Verify the UI looks and functions identical to Phase 1.
- [ ] **Expected Outcome:**
  - [ ] GUI launches without errors and has the same look and functionality as in Phase 1.
  - [ ] All UI elements are created and work correctly.
  - [ ] Start/Stop functionality works as in Phase 1.
- [ ] **Confirmation:** State "Phase 2 completed successfully. All objectives met." if all checkboxes are true. Otherwise, STOP and report issues.

---

## Phase 3: Modifying `api.py` for Detailed Message Logging

**Actions:**

- [ ] **Modify `api.py` for Specific Logging:**

  - [ ] Action: Edit `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py`. Locate the `process_dialog` and `process_text` (if you want to log it too) functions. Add `logger.info()` calls with the `API_REQUEST:`, `API_RESPONSE:`, and `API_ERROR:` prefixes as shown in the plan.
  - [ ] Full Path: `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py`
  - [ ] Code Snippet to add (ensure correct logger object is used):

    ```python
    # In /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py
    # Ensure logger is configured, e.g.:
    # import logging
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # logger = logging.getLogger("nlu-api") # Or your existing FastAPI logger

    # ... inside process_dialog ...
    # logger.info(f"API_REQUEST: /api/dialog | ConvID: {conversation_id} | User: '{request.text}'")
    # ...
    # logger.info(f"API_RESPONSE: /api/dialog | ConvID: {conversation_id} | Bot: '{response_text}'")
    # ...
    # logger.error(f"API_ERROR: /api/dialog | ConvID: {conversation_id} | Error: {e}", exc_info=True)

    # ... inside process_text (optional) ...
    # logger.info(f"API_REQUEST: /api/nlu | User: '{request.text}'")
    # ...
    # logger.info(f"API_RESPONSE: /api/nlu | Intent: {result.get('intent',{}).get('name')}")
    # ...
    # logger.error(f"API_ERROR: /api/nlu | Error: {e}", exc_info=True)
    ```

  - [ ] **Crucial for macOS:** Ensure the logger used (`logger.info`, `logger.error`) in `api.py` actually prints to `sys.stdout` or `sys.stderr` for the `gui_launcher.py` subprocess to capture it. Standard Python `logging` to console handlers will do this. For macOS, flush the output regularly to ensure real-time updates in the UI.

**Objective Check (Phase 3):**

- [ ] **Cursor Action:**
  - [ ] Verify that `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py` has been updated with the new logging lines.
  - [ ] Run the API server _manually_ from a terminal: `python /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/api.py`.
  - [ ] Send a POST request to `/api/dialog` (e.g., `curl http://localhost:8001/api/dialog -X POST -H "Content-Type: application/json" -d '{"text":"test message"}'`). Adjust port if needed.
  - [ ] Send a POST request to `/api/nlu` if you added logging there.
- [ ] **Expected Outcome:**
  - [ ] The console output from running `api.py` manually now includes lines prefixed with `API_REQUEST:`, `API_RESPONSE:`, and `API_ERROR:` (if an error is triggered) for the respective endpoints. Example:
    ```
    2023-10-27 10:00:00,123 - nlu-api - INFO - API_REQUEST: /api/dialog | ConvID: ... | User: 'test message'
    ... other logs ...
    2023-10-27 10:00:00,456 - nlu-api - INFO - API_RESPONSE: /api/dialog | ConvID: ... | Bot: 'Some bot reply'
    ```
- [ ] **Confirmation:** State "Phase 3 completed successfully. All objectives met." if all checkboxes are true. Otherwise, STOP and report issues.

---

## Phase 4: Testing the Complete Application

**Actions:**

- [ ] **Verify All Components Are Working Together:**
  - [ ] Run the application: `python /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py`
  - [ ] Start the API server using the GUI.
  - [ ] Send a request to `/api/dialog` (e.g., `curl http://localhost:8001/api/dialog -X POST -H "Content-Type: application/json" -d '{"text":"testing SOLID principles"}'`). Adjust port if needed.
  - [ ] Observe both log areas in the GUI.
  - [ ] Test clear log buttons.
  - [ ] Test changing the port number in the configuration.
  - [ ] Stop the API server and verify cleanup.
  - [ ] Using macOS Activity Monitor, verify that the Python process for the API server is properly terminated when stopped.

**Objective Check (Phase 4):**

- [ ] **Expected Outcome:**
  - [ ] GUI launches, API starts correctly.
  - [ ] "API Messages Log (Requests/Responses)" area:
    - [ ] Correctly displays lines from `api.py` that are prefixed with `API_REQUEST:` and `API_RESPONSE:`.
    - [ ] The logger's own prefix (e.g., `INFO:nlu-api:`) should ideally be stripped for cleaner display in this log.
  - [ ] "Raw Server Output / Errors" area:
    - [ ] Displays other server output (Uvicorn messages, general logs from `api.py` _not_ matching the API_REQUEST/RESPONSE pattern, and anything sent to stderr).
    - [ ] Lines from stderr or containing "ERROR" should be clearly identifiable.
  - [ ] Clear log buttons function correctly.
  - [ ] Changing the port in configuration works and is applied when API is restarted.
  - [ ] API process correctly terminates when stopped from GUI or when GUI is closed (verify in macOS Activity Monitor).
  - [ ] The UI is Mac-styled with rounded corners and proper spacing.
- [ ] **Confirmation:** State "Phase 4 completed successfully. All objectives met." if all checkboxes are true. Otherwise, STOP and report issues.

---

## Phase 5: macOS-Specific Refinements and Extensions (Optional)

**Actions (Implement as needed):**

- [ ] **Improve macOS Process Management:**

  - [ ] Update the `api_manager.py` to add more robust macOS-specific process handling:

    ```python
    # Add to ApiProcessManager class in /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/api_manager.py
    def _get_child_processes(self, parent_pid):
        """Get all child processes for the given parent PID on macOS"""
        try:
            import subprocess
            # Use macOS-specific 'pgrep' to find child processes
            result = subprocess.run(['pgrep', '-P', str(parent_pid)],
                                   capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return [int(pid) for pid in result.stdout.strip().split('\n')]
            return []
        except Exception:
            return []

    def terminate_with_children(self):
        """Terminate the process and all its children (macOS-specific)"""
        if not self.is_running():
            return False, "API server is not running"

        try:
            # First get all child PIDs
            child_pids = self._get_child_processes(self.process.pid)

            # Try graceful termination first
            os.kill(self.process.pid, signal.SIGTERM)

            # Also terminate any child processes
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass

            # Wait briefly for processes to terminate
            time.sleep(0.5)

            # Check if any processes are still running and force kill if needed
            if self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGKILL)

            for pid in child_pids:
                try:
                    # Check if process still exists
                    os.kill(pid, 0)  # Signal 0 is used to check existence
                    # If we get here, process exists, so force kill
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    # Process is already gone
                    pass

            return True, "API server and all child processes terminated"
        except Exception as e:
            return False, f"Error terminating processes: {e}"
    ```

- [ ] **Add macOS Native Notifications:**

  - [ ] Update NluGuiLauncher to use macOS native notifications:

    ```python
    # Add to NluGuiLauncher class in /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui_launcher.py
    def _send_mac_notification(self, title, message):
        """Send a macOS native notification"""
        try:
            import subprocess
            # Use AppleScript to send a native notification
            script = f'''
            osascript -e 'display notification "{message}" with title "{title}"'
            '''
            subprocess.run(script, shell=True)
        except Exception:
            # Fallback to nicegui notification if native notification fails
            pass

    def _start_api(self):
        """Start the API server with configured parameters"""
        # Get command and environment based on configuration
        command = self.config.get_api_command()
        env = self.config.get_api_env()

        # Start the API process with the configured settings
        success, message = self.api_manager.start(command=command, env=env)

        # Show notification
        ui.notify(message, type='success' if success else 'negative')
        if success:
            self._send_mac_notification("NLU API Started", f"API server started on port {self.config.api_port}")
    ```

- [ ] **Add Auto-Detect API Script:**

  - [ ] Update config.py to automatically discover and select API script if the default isn't found:

    ```python
    # Add to AppConfig class in /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/gui/config.py
    def find_api_script(self):
        """Find API script in the project directory (macOS-specific)"""
        import os
        import glob

        # Start with the current directory
        base_dir = os.path.dirname(os.path.dirname(__file__))

        # List of common API script patterns
        patterns = [
            os.path.join(base_dir, "api.py"),
            os.path.join(base_dir, "app.py"),
            os.path.join(base_dir, "server.py"),
            os.path.join(base_dir, "main.py"),
            os.path.join(base_dir, "*api*.py"),
            os.path.join(base_dir, "*server*.py"),
        ]

        # Check each pattern
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                # Return the first match
                return matches[0]

        # Nothing found
        return None
    ```

- [ ] **Add macOS Dock Icon:**

  - [ ] Create a simple macOS app wrapper (optional):

    ```bash
    # Create a simple script to make a macOS app wrapper
    # Save this as /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/scripts/make_app.sh

    #!/bin/bash

    # Define app name
    APP_NAME="NLU Control"
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gui_launcher.py"

    # Create app directory structure
    mkdir -p "${APP_NAME}.app/Contents/MacOS"
    mkdir -p "${APP_NAME}.app/Contents/Resources"

    # Create launcher script
    cat > "${APP_NAME}.app/Contents/MacOS/launcher" << EOF
    #!/bin/bash
    cd "$(dirname "\$0")/../../.."
    python3 "${SCRIPT_PATH}"
    EOF

    # Make it executable
    chmod +x "${APP_NAME}.app/Contents/MacOS/launcher"

    # Create Info.plist
    cat > "${APP_NAME}.app/Contents/Info.plist" << EOF
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>CFBundleExecutable</key>
        <string>launcher</string>
        <key>CFBundleIconFile</key>
        <string>AppIcon</string>
        <key>CFBundleIdentifier</key>
        <string>com.example.nlucontrol</string>
        <key>CFBundleInfoDictionaryVersion</key>
        <string>6.0</string>
        <key>CFBundleName</key>
        <string>${APP_NAME}</string>
        <key>CFBundlePackageType</key>
        <string>APPL</string>
        <key>CFBundleShortVersionString</key>
        <string>1.0</string>
        <key>CFBundleVersion</key>
        <string>1</string>
        <key>NSHighResolutionCapable</key>
        <true/>
    </dict>
    </plist>
    EOF

    echo "App created at ${APP_NAME}.app"
    ```

**Objective Check (Phase 5):**

- [ ] **Cursor Action:**
  - [ ] If any refinements were implemented, verify their functionality.
  - [ ] Test the improved process management on macOS.
  - [ ] Test macOS native notifications if implemented.
  - [ ] Test the auto-detect API script if implemented.
  - [ ] Test the macOS app wrapper if created.
- [ ] **Expected Outcome:**
  - [ ] Improved process management:
    - [ ] All API child processes are properly terminated.
    - [ ] No orphaned processes when the GUI is closed.
  - [ ] Native notifications:
    - [ ] macOS notifications appear for API start/stop events.
  - [ ] Auto-detect:
    - [ ] If the default API script doesn't exist, an alternative is found.
  - [ ] macOS app wrapper:
    - [ ] The application appears as a regular app with a dock icon.
- [ ] **Confirmation:** State "Phase 5 completed successfully. All implemented objectives met."

---

## SOLID Principles Applied

1. **Single Responsibility Principle (SRP)**

   - `ApiProcessManager`: Responsible only for API process lifecycle.
   - `LogHandler`: Responsible only for log processing.
   - `UiComponents`: Responsible only for UI component creation.
   - `AppConfig`: Responsible only for configuration.

2. **Open/Closed Principle (OCP)**

   - Components are designed to be extended without modification.
   - UI components can be extended with new types of components.
   - Log handler can be extended to support new log formats.

3. **Liskov Substitution Principle (LSP)**

   - Component interfaces (methods, callbacks) are consistent.
   - Different implementations of components could be substituted.

4. **Interface Segregation Principle (ISP)**

   - Components have focused interfaces with specific purposes.
   - Callbacks are separated by responsibility.

5. **Dependency Inversion Principle (DIP)**
   - High-level modules (`NluGuiLauncher`) depend on abstractions.
   - Components communicate through callbacks rather than direct dependencies.

## DRY Principles Applied

1. **Extracted Common UI Creation Logic**

   - UI components are created once and reused.

2. **Centralized Configuration**

   - Configuration is managed in a single place.

3. **Reusable Log Processing**

   - Log parsing and filtering logic is in one place.

4. **Consistent Error Handling**
   - Error handling patterns are consistent across components.

---

**Note:** This implementation is specifically optimized for macOS at `/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot`.
