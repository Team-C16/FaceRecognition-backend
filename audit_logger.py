import os
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file_path=None):
        if log_file_path is None:
            self.log_file_path = os.path.join(os.path.dirname(__file__), "data", "audit_log.json")
        else:
            self.log_file_path = log_file_path
            
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_event(self, event_type, person_name, score=0.0, details=""):
        """
        Logs a security event to the JSON audit file.
        """
        now = datetime.now().isoformat()
        log_entry = {
            "timestamp": now,
            "event_type": event_type,
            "person": person_name,
            "score": round(score, 4),
            "details": details
        }
        
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
            
        logs.append(log_entry)
        
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
        return log_entry
