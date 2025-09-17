from flask import Blueprint

# use top-level templates dir; templates/reviews/*.html
bp = Blueprint("reviews", __name__, url_prefix="/reviews", template_folder="../templates")

from . import routes  # noqa: E402,F401
