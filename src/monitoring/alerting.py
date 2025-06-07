"""
Alerting system for RiskFlow Credit Risk MLOps Pipeline.

Provides configurable alerting for:
- Model performance degradation
- API performance issues
- System resource alerts
- Drift detection alerts
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import time
from collections import defaultdict, deque
from sqlalchemy import text

from config.settings import get_settings
from utils.database import DatabaseManager
from utils.helpers import get_utc_now


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert container with metadata."""
    
    id: Optional[str] = None
    timestamp: datetime = field(default_factory=get_utc_now)
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Alert identification
    alert_type: str = "unknown"
    source: str = "system"
    title: str = ""
    message: str = ""
    
    # Contextual information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Alert management
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    name: str
    description: str
    enabled: bool = True
    
    # Rule conditions
    metric_name: str = ""
    operator: str = ">"  # >, <, >=, <=, ==, !=
    threshold: float = 0.0
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.MEDIUM
    alert_type: str = "threshold"
    
    # Evaluation settings
    evaluation_window: int = 300  # seconds
    consecutive_failures: int = 1
    cooldown_period: int = 3600  # seconds
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    
    # Additional filters
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send alert notification."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test channel connectivity."""
        pass


class LoggingChannel(AlertChannel):
    """Simple logging-based alert channel."""
    
    def __init__(self, log_level: str = "WARNING"):
        self.log_level = log_level.upper()
    
    def send_alert(self, alert: Alert) -> bool:
        """Log alert to console/file."""
        try:
            alert_msg = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            if alert.current_value is not None and alert.threshold_value is not None:
                alert_msg += f" (Current: {alert.current_value}, Threshold: {alert.threshold_value})"
            
            print(f"{alert.timestamp.isoformat()} - ALERT - {alert_msg}")
            return True
            
        except Exception as e:
            print(f"Error sending log alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test logging channel."""
        return True


class EmailChannel(AlertChannel):
    """Email-based alert channel."""
    
    def __init__(self, 
                 smtp_host: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_email: str,
                 to_emails: List[str],
                 use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[RiskFlow Alert] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email alert: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body for alert."""
        severity_color = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107", 
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body>
            <h2 style="color: {severity_color};">RiskFlow Alert - {alert.severity.value.title()}</h2>
            <h3>{alert.title}</h3>
            <p><strong>Message:</strong> {alert.message}</p>
            <p><strong>Timestamp:</strong> {alert.timestamp.isoformat()}</p>
            <p><strong>Source:</strong> {alert.source}</p>
            
            {f'<p><strong>Model:</strong> {alert.model_name} (v{alert.model_version})</p>' if alert.model_name else ''}
            {f'<p><strong>Metric:</strong> {alert.metric_name}</p>' if alert.metric_name else ''}
            {f'<p><strong>Current Value:</strong> {alert.current_value}</p>' if alert.current_value is not None else ''}
            {f'<p><strong>Threshold:</strong> {alert.threshold_value}</p>' if alert.threshold_value is not None else ''}
            
            <hr>
            <p><small>Alert ID: {alert.id}</small></p>
            <p><small>Generated by RiskFlow MLOps Pipeline</small></p>
        </body>
        </html>
        """
    
    def test_connection(self) -> bool:
        """Test email connection."""
        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.quit()
            return True
        except Exception:
            return False


class WebhookChannel(AlertChannel):
    """Webhook-based alert channel (Slack, Teams, etc.)."""
    
    def __init__(self, webhook_url: str, webhook_format: str = "slack"):
        self.webhook_url = webhook_url
        self.webhook_format = webhook_format.lower()
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import requests
            
            if self.webhook_format == "slack":
                payload = self._create_slack_payload(alert)
            else:
                payload = self._create_generic_payload(alert)
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending webhook alert: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack-formatted payload."""
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "warning",
            AlertSeverity.CRITICAL: "danger"
        }
        
        return {
            "text": f"RiskFlow Alert: {alert.title}",
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.title(), "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Message", "value": alert.message, "short": False},
                    {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }]
        }
    
    def _create_generic_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create generic webhook payload."""
        return {
            "alert_id": alert.id,
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
            "source": alert.source,
            "model_name": alert.model_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value
        }
    
    def test_connection(self) -> bool:
        """Test webhook connection."""
        try:
            import requests
            test_payload = {"text": "RiskFlow alert system test"}
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class AlertManager:
    """
    Comprehensive alert management system.
    
    Manages alert rules, evaluation, notification, and lifecycle.
    Thread-safe with background evaluation and notification delivery.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.settings = get_settings()
        self.db_manager = db_manager or DatabaseManager()
        
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_channels: Dict[str, AlertChannel] = {}
        self._alert_history: deque = deque(maxlen=10000)
        
        # Alert evaluation state
        self._rule_failures: Dict[str, int] = defaultdict(int)
        self._last_alert_time: Dict[str, datetime] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        
        # Initialize database tables
        self._init_alert_tables()
        
        # Setup default channels
        self._setup_default_channels()
    
    def _init_alert_tables(self):
        """Initialize alert tables in database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        severity TEXT NOT NULL,
                        status TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        model_name TEXT,
                        model_version TEXT,
                        metric_name TEXT,
                        current_value REAL,
                        threshold_value REAL,
                        metadata TEXT,
                        tags TEXT,
                        acknowledged_by TEXT,
                        acknowledged_at DATETIME,
                        resolved_at DATETIME,
                        suppressed_until DATETIME
                    )
                """))
                
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        enabled BOOLEAN DEFAULT 1,
                        metric_name TEXT,
                        operator TEXT,
                        threshold REAL,
                        severity TEXT,
                        alert_type TEXT,
                        evaluation_window INTEGER,
                        consecutive_failures INTEGER,
                        cooldown_period INTEGER,
                        notification_channels TEXT,
                        filters TEXT,
                        metadata TEXT
                    )
                """))
                
                session.commit()
                
        except Exception as e:
            print(f"Error initializing alert tables: {e}")
    
    def _setup_default_channels(self):
        """Setup default alert channels."""
        # Always have logging channel
        self.add_channel("logging", LoggingChannel())
        
        # Add email channel if configured
        if (hasattr(self.settings, 'SMTP_HOST') and 
            hasattr(self.settings, 'SMTP_USERNAME') and
            hasattr(self.settings, 'ALERT_EMAILS')):
            try:
                email_channel = EmailChannel(
                    smtp_host=self.settings.SMTP_HOST,
                    smtp_port=getattr(self.settings, 'SMTP_PORT', 587),
                    username=self.settings.SMTP_USERNAME,
                    password=getattr(self.settings, 'SMTP_PASSWORD', ''),
                    from_email=getattr(self.settings, 'ALERT_FROM_EMAIL', self.settings.SMTP_USERNAME),
                    to_emails=self.settings.ALERT_EMAILS
                )
                self.add_channel("email", email_channel)
            except Exception as e:
                print(f"Error setting up email channel: {e}")
    
    def add_channel(self, name: str, channel: AlertChannel):
        """Add alert notification channel."""
        with self._lock:
            self._alert_channels[name] = channel
    
    def remove_channel(self, name: str):
        """Remove alert notification channel."""
        with self._lock:
            self._alert_channels.pop(name, None)
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        with self._lock:
            self._alert_rules[rule.name] = rule
            self._persist_rule(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule."""
        with self._lock:
            self._alert_rules.pop(rule_name, None)
            self._delete_rule(rule_name)
    
    def create_alert(self, 
                    title: str,
                    message: str,
                    severity: AlertSeverity = AlertSeverity.MEDIUM,
                    alert_type: str = "manual",
                    source: str = "user",
                    **kwargs) -> Alert:
        """Create and fire an alert."""
        
        alert = Alert(
            id=self._generate_alert_id(),
            title=title,
            message=message,
            severity=severity,
            alert_type=alert_type,
            source=source,
            **kwargs
        )
        
        return self.fire_alert(alert)
    
    def fire_alert(self, alert: Alert) -> Alert:
        """Fire an alert and send notifications."""
        
        # Generate ID if not provided
        if not alert.id:
            alert.id = self._generate_alert_id()
        
        with self._lock:
            # Store alert
            self._active_alerts[alert.id] = alert
            self._alert_history.append(alert)
            
            # Persist to database
            self._persist_alert(alert)
            
            # Send notifications
            self._send_notifications(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = get_utc_now()
                
                self._persist_alert(alert)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = get_utc_now()
                
                self._persist_alert(alert)
                
                # Remove from active alerts
                del self._active_alerts[alert_id]
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress an alert for a duration."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.suppressed_until = get_utc_now() + timedelta(minutes=duration_minutes)
                
                self._persist_alert(alert)
    
    def evaluate_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Evaluate a metric against alert rules."""
        
        with self._lock:
            for rule_name, rule in self._alert_rules.items():
                if not rule.enabled or rule.metric_name != metric_name:
                    continue
                
                # Check if metric violates threshold
                violation = self._check_threshold_violation(rule, value)
                
                if violation:
                    self._rule_failures[rule_name] += 1
                    
                    # Check if we should fire alert
                    if (self._rule_failures[rule_name] >= rule.consecutive_failures and
                        self._check_cooldown(rule_name, rule.cooldown_period)):
                        
                        # Create alert
                        alert = Alert(
                            title=f"Metric Alert: {metric_name}",
                            message=f"Metric {metric_name} {rule.operator} {rule.threshold} (current: {value})",
                            severity=rule.severity,
                            alert_type=rule.alert_type,
                            source="metric_evaluation",
                            metric_name=metric_name,
                            current_value=value,
                            threshold_value=rule.threshold,
                            metadata=metadata or {}
                        )
                        
                        self.fire_alert(alert)
                        self._last_alert_time[rule_name] = get_utc_now()
                        self._rule_failures[rule_name] = 0
                
                else:
                    # Reset failure count on successful evaluation
                    self._rule_failures[rule_name] = 0
    
    def _check_threshold_violation(self, rule: AlertRule, value: float) -> bool:
        """Check if value violates rule threshold."""
        operators = {
            '>': lambda x, t: x > t,
            '<': lambda x, t: x < t,
            '>=': lambda x, t: x >= t,
            '<=': lambda x, t: x <= t,
            '==': lambda x, t: x == t,
            '!=': lambda x, t: x != t
        }
        
        operator_func = operators.get(rule.operator)
        if not operator_func:
            return False
        
        return operator_func(value, rule.threshold)
    
    def _check_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """Check if rule is in cooldown period."""
        if rule_name not in self._last_alert_time:
            return True
        
        last_alert = self._last_alert_time[rule_name]
        return (get_utc_now() - last_alert).total_seconds() > cooldown_seconds
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels."""
        
        # Get notification channels from rule or use all channels
        channels_to_use = ["logging"]  # Always log
        
        # Find matching rule for additional channels
        for rule in self._alert_rules.values():
            if (rule.metric_name == alert.metric_name or 
                rule.alert_type == alert.alert_type):
                channels_to_use.extend(rule.notification_channels)
                break
        
        # Send notifications
        for channel_name in set(channels_to_use):
            if channel_name in self._alert_channels:
                try:
                    self._alert_channels[channel_name].send_alert(alert)
                except Exception as e:
                    print(f"Error sending alert via {channel_name}: {e}")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT OR REPLACE INTO alerts (
                        id, timestamp, severity, status, alert_type, source, title, message,
                        model_name, model_version, metric_name, current_value, threshold_value,
                        metadata, tags, acknowledged_by, acknowledged_at, resolved_at, suppressed_until
                    ) VALUES (
                        :id, :timestamp, :severity, :status, :alert_type, :source, :title, :message,
                        :model_name, :model_version, :metric_name, :current_value, :threshold_value,
                        :metadata, :tags, :acknowledged_by, :acknowledged_at, :resolved_at, :suppressed_until
                    )
                """), {
                    'id': alert.id,
                    'timestamp': alert.timestamp,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'alert_type': alert.alert_type,
                    'source': alert.source,
                    'title': alert.title,
                    'message': alert.message,
                    'model_name': alert.model_name,
                    'model_version': alert.model_version,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'metadata': json.dumps(alert.metadata),
                    'tags': json.dumps(alert.tags),
                    'acknowledged_by': alert.acknowledged_by,
                    'acknowledged_at': alert.acknowledged_at,
                    'resolved_at': alert.resolved_at,
                    'suppressed_until': alert.suppressed_until
                })
                session.commit()
                
        except Exception as e:
            print(f"Error persisting alert: {e}")
    
    def _persist_rule(self, rule: AlertRule):
        """Persist alert rule to database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT OR REPLACE INTO alert_rules (
                        name, description, enabled, metric_name, operator, threshold,
                        severity, alert_type, evaluation_window, consecutive_failures,
                        cooldown_period, notification_channels, filters, metadata
                    ) VALUES (
                        :name, :description, :enabled, :metric_name, :operator, :threshold,
                        :severity, :alert_type, :evaluation_window, :consecutive_failures,
                        :cooldown_period, :notification_channels, :filters, :metadata
                    )
                """), {
                    'name': rule.name,
                    'description': rule.description,
                    'enabled': rule.enabled,
                    'metric_name': rule.metric_name,
                    'operator': rule.operator,
                    'threshold': rule.threshold,
                    'severity': rule.severity.value,
                    'alert_type': rule.alert_type,
                    'evaluation_window': rule.evaluation_window,
                    'consecutive_failures': rule.consecutive_failures,
                    'cooldown_period': rule.cooldown_period,
                    'notification_channels': json.dumps(rule.notification_channels),
                    'filters': json.dumps(rule.filters),
                    'metadata': json.dumps(rule.metadata)
                })
                session.commit()
                
        except Exception as e:
            print(f"Error persisting rule: {e}")
    
    def _delete_rule(self, rule_name: str):
        """Delete alert rule from database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("DELETE FROM alert_rules WHERE name = :name"), {'name': rule_name})
                session.commit()
        except Exception as e:
            print(f"Error deleting rule: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history from database."""
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM alerts 
                    WHERE timestamp >= :start_time
                    ORDER BY timestamp DESC
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            print(f"Error getting alert history: {e}")
            return []
    
    def get_alert_rules(self) -> List[AlertRule]:
        """Get list of alert rules."""
        with self._lock:
            return list(self._alert_rules.values()) 