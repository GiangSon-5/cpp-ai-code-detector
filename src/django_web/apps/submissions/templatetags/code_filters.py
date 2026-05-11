"""
submissions/templatetags/code_filters.py — Custom Django template filters.
"""

from django import template

register = template.Library()


@register.filter(name="splitlines")
def splitlines(value):
    """Split a string into lines. Used in result.html to display source code with line numbers."""
    if not value:
        return []
    return value.splitlines()
