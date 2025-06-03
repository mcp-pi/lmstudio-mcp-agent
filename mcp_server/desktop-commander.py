#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import subprocess
import shutil
import stat
import base64
import mimetypes
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import tempfile
import threading
import queue
import uuid
from urllib.parse import urlparse
import requests

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    "Desktop_Commander",
    instructions="A comprehensive desktop commander that provides file system operations, terminal execution, code search, and text editing capabilities.",
    host="0.0.0.0",
    port=1109,
)

# Configuration
CONFIG_DIR = Path.home() / '.claude-server-commander'
CONFIG_FILE = CONFIG_DIR / 'config.json'
VERSION = '0.2.2-python'

@dataclass
class ServerConfig:
    blocked_commands: List[str]
    default_shell: str
    allowed_directories: List[str]
    telemetry_enabled: bool
    file_write_line_limit: int
    file_read_line_limit: int
    version: str

@dataclass
class TerminalSession:
    pid: int
    process: subprocess.Popen
    last_output: str
    is_blocked: bool
    start_time: datetime
    output_queue: queue.Queue

@dataclass
class SearchResult:
    file: str
    line: int
    match: str

class ConfigManager:
    def __init__(self):
        self.config_path = CONFIG_FILE
        self.config = {}
        self.initialized = False

    async def init(self):
        if self.initialized:
            return
        
        try:
            # Ensure config directory exists
            CONFIG_DIR.mkdir(exist_ok=True)
            
            # Load or create config
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                await self._save_config()
            
            self.config['version'] = VERSION
            self.initialized = True
        except Exception as e:
            print(f"Failed to initialize config: {e}", file=sys.stderr)
            self.config = self._get_default_config()
            self.initialized = True

    def _get_default_config(self) -> dict:
        return {
            "blocked_commands": [
                "mkfs", "format", "mount", "umount", "fdisk", "dd", "parted", "diskpart",
                "sudo", "su", "passwd", "adduser", "useradd", "usermod", "groupadd", 
                "chsh", "visudo", "shutdown", "reboot", "halt", "poweroff", "init",
                "iptables", "firewall", "netsh", "sfc", "bcdedit", "reg", "net", 
                "sc", "runas", "cipher", "takeown"
            ],
            "default_shell": "powershell.exe" if os.name == 'nt' else "bash",
            "allowed_directories": [],
            "telemetry_enabled": True,
            "file_write_line_limit": 50,
            "file_read_line_limit": 1000
        }

    async def _save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}", file=sys.stderr)

    async def get_config(self) -> dict:
        await self.init()
        return self.config.copy()

    async def get_value(self, key: str) -> Any:
        await self.init()
        return self.config.get(key)

    async def set_value(self, key: str, value: Any):
        await self.init()
        self.config[key] = value
        await self._save_config()

class TerminalManager:
    def __init__(self):
        self.sessions: Dict[int, TerminalSession] = {}
        self.completed_sessions: Dict[int, dict] = {}

    def _read_output_thread(self, process: subprocess.Popen, output_queue: queue.Queue):
        """Thread function to read process output"""
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_queue.put(('stdout', output))
                
                error = process.stderr.readline()
                if error:
                    output_queue.put(('stderr', error))
        except Exception as e:
            output_queue.put(('error', str(e)))

    async def execute_command(self, command: str, timeout_ms: int = 30000, shell: Optional[str] = None) -> dict:
        try:
            # Use provided shell or default
            if not shell:
                shell_cmd = await config_manager.get_value('default_shell') or ('cmd' if os.name == 'nt' else 'bash')
            else:
                shell_cmd = shell

            # Start process
            if os.name == 'nt':
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid
                )

            if not process.pid:
                return {
                    "pid": -1,
                    "output": "Error: Failed to get process ID. The command could not be executed.",
                    "is_blocked": False
                }

            # Create session
            output_queue = queue.Queue()
            session = TerminalSession(
                pid=process.pid,
                process=process,
                last_output="",
                is_blocked=False,
                start_time=datetime.now(),
                output_queue=output_queue
            )
            
            self.sessions[process.pid] = session

            # Start output reading thread
            output_thread = threading.Thread(
                target=self._read_output_thread,
                args=(process, output_queue)
            )
            output_thread.daemon = True
            output_thread.start()

            # Wait for initial output or timeout
            start_time = time.time()
            timeout_seconds = timeout_ms / 1000
            output = ""
            
            while time.time() - start_time < timeout_seconds:
                try:
                    msg_type, msg_content = output_queue.get_nowait()
                    if msg_type in ['stdout', 'stderr']:
                        output += msg_content
                        session.last_output += msg_content
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.1)

            # Check if process is still running
            is_blocked = process.poll() is None

            if not is_blocked:
                # Process completed
                return_code = process.poll()
                self.completed_sessions[process.pid] = {
                    "pid": process.pid,
                    "output": output + session.last_output,
                    "exit_code": return_code,
                    "start_time": session.start_time,
                    "end_time": datetime.now()
                }
                del self.sessions[process.pid]
            else:
                session.is_blocked = True

            return {
                "pid": process.pid,
                "output": output,
                "is_blocked": is_blocked
            }

        except Exception as e:
            return {
                "pid": -1,
                "output": f"Error executing command: {str(e)}",
                "is_blocked": False
            }

    def get_new_output(self, pid: int) -> Optional[str]:
        session = self.sessions.get(pid)
        if session:
            # Get any new output from queue
            new_output = ""
            try:
                while True:
                    msg_type, msg_content = session.output_queue.get_nowait()
                    if msg_type in ['stdout', 'stderr']:
                        new_output += msg_content
            except queue.Empty:
                pass
            
            # Check if process completed
            if session.process.poll() is not None and pid in self.sessions:
                return_code = session.process.poll()
                runtime = (datetime.now() - session.start_time).total_seconds()
                completion_msg = f"\nProcess completed with exit code {return_code}\nRuntime: {runtime:.1f}s"
                
                self.completed_sessions[pid] = {
                    "pid": pid,
                    "output": session.last_output + new_output,
                    "exit_code": return_code,
                    "start_time": session.start_time,
                    "end_time": datetime.now()
                }
                del self.sessions[pid]
                return new_output + completion_msg
            
            return new_output if new_output else None

        # Check completed sessions
        completed = self.completed_sessions.get(pid)
        if completed:
            runtime = (completed["end_time"] - completed["start_time"]).total_seconds()
            return f"Process completed with exit code {completed['exit_code']}\nRuntime: {runtime:.1f}s\nFinal output:\n{completed['output']}"

        return None

    def force_terminate(self, pid: int) -> bool:
        session = self.sessions.get(pid)
        if not session:
            return False

        try:
            if os.name == 'nt':
                # Windows
                session.process.terminate()
            else:
                # Unix-like
                os.killpg(os.getpgid(session.process.pid), signal.SIGTERM)
            
            # Wait a bit then force kill if still running
            time.sleep(1)
            if session.process.poll() is None:
                if os.name == 'nt':
                    session.process.kill()
                else:
                    os.killpg(os.getpgid(session.process.pid), signal.SIGKILL)
            
            del self.sessions[pid]
            return True
        except Exception as e:
            print(f"Error terminating process {pid}: {e}", file=sys.stderr)
            return False

    def list_active_sessions(self) -> List[dict]:
        now = datetime.now()
        return [
            {
                "pid": session.pid,
                "is_blocked": session.is_blocked,
                "runtime": int((now - session.start_time).total_seconds() * 1000)
            }
            for session in self.sessions.values()
        ]

class CommandManager:
    async def validate_command(self, command: str) -> bool:
        try:
            config = await config_manager.get_config()
            blocked_commands = config.get('blocked_commands', [])
            
            # Extract base command
            base_command = command.split()[0].lower().strip()
            
            return base_command not in blocked_commands
        except Exception:
            return True  # Default to allowing on error

class FileSystemManager:
    async def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories"""
        try:
            config = await config_manager.get_config()
            allowed_dirs = config.get('allowed_directories', [])
            
            # If no restrictions or root allowed, allow all
            if not allowed_dirs or '/' in allowed_dirs:
                return True
            
            path_obj = Path(path).resolve()
            
            for allowed_dir in allowed_dirs:
                allowed_path = Path(allowed_dir).resolve()
                try:
                    path_obj.relative_to(allowed_path)
                    return True
                except ValueError:
                    continue
            
            return False
        except Exception:
            return True  # Default to allowing on error

    async def validate_path(self, path: str) -> str:
        """Validate and normalize path"""
        # Expand user home directory
        if path.startswith('~'):
            path = str(Path(path).expanduser())
        
        # Convert to absolute path
        abs_path = str(Path(path).resolve())
        
        # Check if allowed
        if not await self._is_path_allowed(abs_path):
            raise PermissionError(f"Path not allowed: {path}")
        
        return abs_path

    async def read_file(self, file_path: str, is_url: bool = False, offset: int = 0, length: int = 1000) -> dict:
        """Read file content from filesystem or URL"""
        if is_url:
            return await self._read_file_from_url(file_path)
        
        validated_path = await self.validate_path(file_path)
        path_obj = Path(validated_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        mime_type = mime_type or 'text/plain'
        is_image = mime_type.startswith('image/')
        
        if is_image:
            # Read as binary for images
            with open(path_obj, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            return {
                "content": content,
                "mime_type": mime_type,
                "is_image": True
            }
        else:
            # Read as text with line-based offset/length
            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines = len(lines)
                start_line = min(offset, total_lines)
                end_line = min(start_line + length, total_lines)
                
                if start_line == total_lines and offset > 0:
                    # Show last few lines instead
                    last_lines_count = min(10, total_lines)
                    start_line = max(0, total_lines - last_lines_count)
                    end_line = total_lines
                
                selected_lines = lines[start_line:end_line]
                content = ''.join(selected_lines)
                
                # Add informational message for partial reads
                if offset > 0 or end_line < total_lines:
                    if offset >= total_lines and total_lines > 0:
                        content = f"[NOTICE: Offset {offset} exceeds file length ({total_lines} lines). Showing last {end_line - start_line} lines instead.]\n\n{content}"
                    else:
                        content = f"[Reading {end_line - start_line} lines from line {start_line} of {total_lines} total lines]\n\n{content}"
                
                return {
                    "content": content,
                    "mime_type": mime_type,
                    "is_image": False
                }
            except UnicodeDecodeError:
                # Fallback to binary
                with open(path_obj, 'rb') as f:
                    content = f"Binary file content (base64 encoded):\n{base64.b64encode(f.read()).decode('utf-8')}"
                return {
                    "content": content,
                    "mime_type": 'text/plain',
                    "is_image": False
                }

    async def _read_file_from_url(self, url: str) -> dict:
        """Read file content from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', 'text/plain')
            is_image = content_type.startswith('image/')
            
            if is_image:
                content = base64.b64encode(response.content).decode('utf-8')
            else:
                content = response.text
            
            return {
                "content": content,
                "mime_type": content_type,
                "is_image": is_image
            }
        except Exception as e:
            raise Exception(f"Failed to fetch URL: {str(e)}")

    async def write_file(self, file_path: str, content: str, mode: str = 'rewrite'):
        """Write content to file"""
        validated_path = await self.validate_path(file_path)
        path_obj = Path(validated_path)
        
        # Create parent directories if needed
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == 'append':
            with open(path_obj, 'a', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path_obj, 'w', encoding='utf-8') as f:
                f.write(content)

    async def list_directory(self, dir_path: str) -> List[str]:
        """List directory contents"""
        validated_path = await self.validate_path(dir_path)
        path_obj = Path(validated_path)
        
        if not path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        entries = []
        for item in path_obj.iterdir():
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{prefix} {item.name}")
        
        return sorted(entries)

    async def create_directory(self, dir_path: str):
        """Create directory"""
        validated_path = await self.validate_path(dir_path)
        Path(validated_path).mkdir(parents=True, exist_ok=True)

    async def move_file(self, source_path: str, dest_path: str):
        """Move/rename file or directory"""
        validated_source = await self.validate_path(source_path)
        validated_dest = await self.validate_path(dest_path)
        
        shutil.move(validated_source, validated_dest)

    async def search_files(self, root_path: str, pattern: str) -> List[str]:
        """Search for files by name pattern"""
        validated_path = await self.validate_path(root_path)
        root_obj = Path(validated_path)
        
        results = []
        
        def search_recursive(path: Path):
            try:
                for item in path.iterdir():
                    if pattern.lower() in item.name.lower():
                        results.append(str(item))
                    
                    if item.is_dir():
                        search_recursive(item)
            except (PermissionError, OSError):
                pass
        
        search_recursive(root_obj)
        return results

    async def get_file_info(self, file_path: str) -> dict:
        """Get file information"""
        validated_path = await self.validate_path(file_path)
        path_obj = Path(validated_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat_info = path_obj.stat()
        
        info = {
            "size": stat_info.st_size,
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            "is_directory": path_obj.is_dir(),
            "is_file": path_obj.is_file(),
            "permissions": oct(stat_info.st_mode)[-3:],
        }
        
        # Add line count for text files
        if path_obj.is_file() and stat_info.st_size < 10 * 1024 * 1024:  # < 10MB
            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    info["line_count"] = len(lines)
                    info["last_line"] = len(lines) - 1
                    info["append_position"] = len(lines)
            except (UnicodeDecodeError, PermissionError):
                pass
        
        return info

    async def search_code(self, root_path: str, pattern: str, file_pattern: str = None, 
                         ignore_case: bool = True, max_results: int = 1000) -> List[SearchResult]:
        """Search for text patterns in files"""
        validated_path = await self.validate_path(root_path)
        root_obj = Path(validated_path)
        
        results = []
        regex_flags = re.IGNORECASE if ignore_case else 0
        search_regex = re.compile(pattern, regex_flags)
        
        def should_include_file(file_path: Path) -> bool:
            if file_pattern:
                import fnmatch
                return fnmatch.fnmatch(file_path.name, file_pattern)
            return True
        
        def search_in_file(file_path: Path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if search_regex.search(line):
                            results.append(SearchResult(
                                file=str(file_path),
                                line=line_num,
                                match=line.strip()
                            ))
                            if len(results) >= max_results:
                                return
            except (UnicodeDecodeError, PermissionError, OSError):
                pass
        
        def search_recursive(path: Path):
            if len(results) >= max_results:
                return
            
            try:
                for item in path.iterdir():
                    if len(results) >= max_results:
                        break
                    
                    if item.is_file() and should_include_file(item):
                        search_in_file(item)
                    elif item.is_dir() and not item.name.startswith('.'):
                        search_recursive(item)
            except (PermissionError, OSError):
                pass
        
        search_recursive(root_obj)
        return results

    async def edit_block(self, file_path: str, old_string: str, new_string: str, expected_replacements: int = 1) -> dict:
        """Edit text in file with search/replace"""
        if not old_string:
            raise ValueError("Empty search strings are not allowed")
        
        validated_path = await self.validate_path(file_path)
        
        # Read current content
        file_result = await self.read_file(validated_path)
        content = file_result["content"]
        
        # Count occurrences
        count = content.count(old_string)
        
        if count == expected_replacements:
            # Perform replacement
            if expected_replacements == 1:
                new_content = content.replace(old_string, new_string, 1)
            else:
                new_content = content.replace(old_string, new_string)
            
            await self.write_file(validated_path, new_content)
            
            return {
                "success": True,
                "message": f"Successfully applied {expected_replacements} edit{'s' if expected_replacements > 1 else ''} to {file_path}",
                "replacements_made": expected_replacements
            }
        elif count > 0:
            return {
                "success": False,
                "message": f"Expected {expected_replacements} occurrences but found {count} in {file_path}. Set expected_replacements to {count} to replace all occurrences.",
                "found_count": count
            }
        else:
            # Try fuzzy matching (simplified)
            if old_string.lower() in content.lower():
                return {
                    "success": False,
                    "message": f"Exact match not found, but similar text exists. Please check the exact text in the file.",
                    "found_count": 0
                }
            else:
                return {
                    "success": False,
                    "message": f"Search content not found in {file_path}",
                    "found_count": 0
                }

# Global instances
config_manager = ConfigManager()
terminal_manager = TerminalManager()
command_manager = CommandManager()
filesystem_manager = FileSystemManager()

# MCP Tools

@mcp.tool()
async def get_config() -> str:
    """Get the complete server configuration as JSON."""
    try:
        config = await config_manager.get_config()
        return f"Current configuration:\n{json.dumps(config, indent=2)}"
    except Exception as e:
        return f"Error getting configuration: {str(e)}"

@mcp.tool()
async def set_config_value(key: str, value: Any) -> str:
    """Set a specific configuration value by key."""
    try:
        # Parse string values that should be arrays
        if key in ['allowed_directories', 'blocked_commands'] and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                if not value.startswith('['):
                    value = [value]
        
        await config_manager.set_value(key, value)
        updated_config = await config_manager.get_config()
        
        return f"Successfully set {key} to {json.dumps(value, indent=2)}\n\nUpdated configuration:\n{json.dumps(updated_config, indent=2)}"
    except Exception as e:
        return f"Error setting value: {str(e)}"

@mcp.tool()
async def read_file(path: str, isUrl: bool = False, offset: int = 0, length: int = 1000) -> str:
    """Read the contents of a file from the file system or a URL."""
    try:
        file_result = await filesystem_manager.read_file(path, isUrl, offset, length)
        
        if file_result["is_image"]:
            return f"Image file: {path} ({file_result['mime_type']})\nBase64 content: {file_result['content'][:100]}..."
        else:
            return file_result["content"]
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
async def write_file(path: str, content: str, mode: str = 'rewrite') -> str:
    """Write or append to file contents."""
    try:
        config = await config_manager.get_config()
        max_lines = config.get('file_write_line_limit', 50)
        
        line_count = len(content.split('\n'))
        warning = ""
        
        if line_count > max_lines:
            warning = f"\n\nWARNING: Content has {line_count} lines (maximum: {max_lines}). Consider splitting into smaller chunks."
        
        await filesystem_manager.write_file(path, content, mode)
        
        mode_message = 'appended to' if mode == 'append' else 'wrote to'
        return f"Successfully {mode_message} {path} ({line_count} lines){warning}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
async def create_directory(path: str) -> str:
    """Create a new directory or ensure a directory exists."""
    try:
        await filesystem_manager.create_directory(path)
        return f"Successfully created directory {path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

@mcp.tool()
async def list_directory(path: str) -> str:
    """Get a detailed listing of all files and directories in a specified path."""
    try:
        entries = await filesystem_manager.list_directory(path)
        return '\n'.join(entries)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@mcp.tool()
async def move_file(source: str, destination: str) -> str:
    """Move or rename files and directories."""
    try:
        await filesystem_manager.move_file(source, destination)
        return f"Successfully moved {source} to {destination}"
    except Exception as e:
        return f"Error moving file: {str(e)}"

@mcp.tool()
async def search_files(path: str, pattern: str) -> str:
    """Find files by name using substring matching."""
    try:
        results = await filesystem_manager.search_files(path, pattern)
        if not results:
            return "No matches found"
        return '\n'.join(results)
    except Exception as e:
        return f"Error searching files: {str(e)}"

@mcp.tool()
async def search_code(path: str, pattern: str, filePattern: str = None, ignoreCase: bool = True, maxResults: int = 1000) -> str:
    """Search for text/code patterns within file contents."""
    try:
        results = await filesystem_manager.search_code(path, pattern, filePattern, ignoreCase, maxResults)
        
        if not results:
            return "No matches found"
        
        # Format results
        formatted_results = ""
        current_file = ""
        
        for result in results:
            if result.file != current_file:
                formatted_results += f"\n{result.file}:\n"
                current_file = result.file
            formatted_results += f"  {result.line}: {result.match}\n"
        
        return formatted_results.strip()
    except Exception as e:
        return f"Error searching code: {str(e)}"

@mcp.tool()
async def get_file_info(path: str) -> str:
    """Retrieve detailed metadata about a file or directory."""
    try:
        info = await filesystem_manager.get_file_info(path)
        return '\n'.join(f"{key}: {value}" for key, value in info.items())
    except Exception as e:
        return f"Error getting file info: {str(e)}"

@mcp.tool()
async def edit_block(file_path: str, old_string: str, new_string: str, expected_replacements: int = 1) -> str:
    """Apply surgical text replacements to files."""
    try:
        result = await filesystem_manager.edit_block(file_path, old_string, new_string, expected_replacements)
        return result["message"]
    except Exception as e:
        return f"Error editing file: {str(e)}"

@mcp.tool()
async def execute_command(command: str, timeout_ms: int = 30000, shell: str = None) -> str:
    """Execute a terminal command with timeout."""
    try:
        # Validate command
        if not await command_manager.validate_command(command):
            return f"Error: Command not allowed: {command}"
        
        result = await terminal_manager.execute_command(command, timeout_ms, shell)
        
        if result["pid"] == -1:
            return result["output"]
        
        message = f"Command started with PID {result['pid']}\nInitial output:\n{result['output']}"
        if result["is_blocked"]:
            message += "\nCommand is still running. Use read_output to get more output."
        
        return message
    except Exception as e:
        return f"Error executing command: {str(e)}"

@mcp.tool()
async def read_output(pid: int, timeout_ms: int = 5000) -> str:
    """Read new output from a running terminal session."""
    try:
        # Wait for output with simple polling
        start_time = time.time()
        timeout_seconds = timeout_ms / 1000
        
        while time.time() - start_time < timeout_seconds:
            output = terminal_manager.get_new_output(pid)
            if output:
                return output
            await asyncio.sleep(0.1)
        
        # Check one more time after timeout
        output = terminal_manager.get_new_output(pid)
        return output or f"No new output available for PID {pid} (timeout reached)"
    except Exception as e:
        return f"Error reading output: {str(e)}"

@mcp.tool()
async def force_terminate(pid: int) -> str:
    """Force terminate a running terminal session."""
    try:
        success = terminal_manager.force_terminate(pid)
        if success:
            return f"Successfully initiated termination of session {pid}"
        else:
            return f"No active session found for PID {pid}"
    except Exception as e:
        return f"Error terminating session: {str(e)}"

@mcp.tool()
async def list_sessions() -> str:
    """List all active terminal sessions."""
    try:
        sessions = terminal_manager.list_active_sessions()
        if not sessions:
            return "No active sessions"
        
        return '\n'.join(
            f"PID: {s['pid']}, Blocked: {s['is_blocked']}, Runtime: {s['runtime']/1000:.1f}s"
            for s in sessions
        )
    except Exception as e:
        return f"Error listing sessions: {str(e)}"

@mcp.tool()
async def list_processes() -> str:
    """List all running processes."""
    try:
        if os.name == 'nt':
            # Windows
            result = subprocess.run(['tasklist'], capture_output=True, text=True)
        else:
            # Unix-like
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error listing processes: {result.stderr}"
    except Exception as e:
        return f"Error listing processes: {str(e)}"

@mcp.tool()
async def kill_process(pid: int) -> str:
    """Terminate a running process by PID."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
        else:
            os.kill(pid, signal.SIGTERM)
        
        return f"Successfully terminated process {pid}"
    except subprocess.CalledProcessError as e:
        return f"Error killing process: Command failed with return code {e.returncode}"
    except ProcessLookupError:
        return f"Process {pid} not found"
    except PermissionError:
        return f"Permission denied to kill process {pid}"
    except Exception as e:
        return f"Error killing process: {str(e)}"

def main():
    """Entry point when package is installed via pip"""
    print("Starting Desktop Commander MCP Server", file=sys.stderr)
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()    