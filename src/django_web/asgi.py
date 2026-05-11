"""ASGI config for Django Web."""

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "src.django_web.settings")

from django.core.asgi import get_asgi_application
application = get_asgi_application()
