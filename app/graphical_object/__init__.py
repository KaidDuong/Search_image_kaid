# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from flask import Blueprint

blueprint = Blueprint(
    'graphical_object_blueprint',
    __name__,
    url_prefix='',
    template_folder='templates',
    static_folder='static',
    static_url_path = ""
)
