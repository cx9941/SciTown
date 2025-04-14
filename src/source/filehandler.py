import json
import os
from datetime import datetime
from typing import Union

class FileHandler:
    """Handler for file operations supporting both JSON and text-based logging.
    
    Args:
        file_path (Union[bool, str]): Path to the log file or boolean flag
    """

    def __init__(self, file_path: Union[bool, str]):
        self._initialize_path(file_path)
        
    def _initialize_path(self, file_path: Union[bool, str]):
        if file_path is True:  # File path is boolean True
            self._path = os.path.join(os.curdir, "logs.txt")
        
        elif isinstance(file_path, str):  # File path is a string
            if file_path.endswith((".json", ".txt")):
                self._path = file_path  # No modification if the file ends with .json or .txt
            else:
                self._path = file_path + ".txt"  # Append .txt if the file doesn't end with .json or .txt
        
        else:
            raise ValueError("file_path must be a string or boolean.")  # Handle the case where file_path isn't valid
        
    def log(self, **kwargs):
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"timestamp": now, **kwargs}

            if self._path.endswith(".json"):
                # Append log in JSON format
                with open(self._path, "a", encoding="utf-8") as file:
                    # If the file is empty, start with a list; else, append to it
                    try:
                        # Try reading existing content to avoid overwriting
                        with open(self._path, "r", encoding="utf-8") as read_file:
                            existing_data = json.load(read_file)
                            existing_data.append(log_entry)
                    except (json.JSONDecodeError, FileNotFoundError):
                        # If no valid JSON or file doesn't exist, start with an empty list
                        existing_data = [log_entry]
                    
                    with open(self._path, "w", encoding="utf-8") as write_file:
                        json.dump(existing_data, write_file, indent=4, ensure_ascii=False)
                        write_file.write("\n")
            
            else:
                # Append log in plain text format
                message = f"{now}: " + ", ".join([f"{key}=\"{value}\"" for key, value in kwargs.items()]) + "\n"
                with open(self._path, "a", encoding="utf-8") as file:
                    file.write(message)

        except Exception as e:
            raise ValueError(f"Failed to log message: {str(e)}")