"""WSGI config for Django Web."""

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "src.django_web.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
