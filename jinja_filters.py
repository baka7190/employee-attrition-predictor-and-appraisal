# jinja_filters.py
from __future__ import annotations
from enum import Enum
from datetime import datetime
from flask import Flask

def register_filters(app: Flask) -> None:
    """
    Register custom Jinja filters on the given Flask app.
    Usage in templates:
        {{ some_enum | label }}
        {{ some_datetime | datetime }}
    """

    @app.template_filter("label")
    def enum_label(value):
        """
        Convert an Enum or string like 'SUBMITTED_TO_MANAGER' into
        'Submitted To Manager'. Works with Enum.name or Enum.value.
        """
        if value is None:
            return ""
        if isinstance(value, Enum):
            raw = getattr(value, "value", None) or getattr(value, "name", str(value))
        else:
            raw = str(value)
        return raw.replace("_", " ").title()

    @app.template_filter("datetime")
    def format_datetime(value, fmt: str = "%Y-%m-%d %H:%M"):
        """
        Render datetime (aware or naive) using a format string.
        If value is falsy or not a datetime, returns empty string.
        """
        if isinstance(value, datetime):
            return value.strftime(fmt)
        # sometimes you pass (updated_at or created_at); if it's not a datetime, bail gracefully
        return ""
