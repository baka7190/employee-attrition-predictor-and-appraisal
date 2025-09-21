from flask import (
    Flask, render_template, render_template_string, request, session,
    redirect, url_for, flash, send_file, jsonify, Blueprint, has_request_context
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from sqlalchemy import or_, func
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.routing import BuildError
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

import os, glob, json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date
from io import BytesIO

# Optional mail
try:
    from flask_mail import Mail, Message
    MAIL_AVAILABLE = True
except Exception:
    Mail = Message = None
    MAIL_AVAILABLE = False

from extensions import db, migrate
from models import (
    Role, User, Prediction, PerformanceReview, ReviewStatus, Criteria,
    Appraisal, CycleEnrollment, RetentionPlan
)
from pm_reviews import bp as reviews_bp
from appraisal_routes import appraisal_bp
from forms import LoginForm

# ---------------- Utilities (roles, helpers) ----------------
def _role_to_string(r) -> str:
    if r is None: return ""
    if isinstance(r, str): return r
    if hasattr(r, "value") and isinstance(getattr(r, "value"), str): return r.value
    if hasattr(r, "name") and isinstance(getattr(r, "name"), str): return r.name
    for attr in ("name", "code", "title", "label", "role"):
        v = getattr(r, attr, None)
        if isinstance(v, str): return v
    return str(r)

def _current_role() -> str:
    s = ""
    try:
        if current_user and hasattr(current_user, "role"):
            s = _role_to_string(getattr(current_user, "role", None))
    except Exception:
        s = ""
    if not s and has_request_context():
        s = session.get("role", "")
    return (s or "").strip().upper()

def get_role(user) -> str:
    try:
        val = getattr(user, "role", None)
        if isinstance(val, Role): return (val.value or "").upper()
        return (str(val) or "").upper()
    except Exception:
        return ""

def _verify_password(stored_hash_or_plain: str, provided: str) -> bool:
    """
    Accepts both proper Werkzeug hashes and legacy plaintext.
    This makes old databases (that stored plaintext) still log in.
    """
    if not stored_hash_or_plain:
        return False
    try:
        # Looks like a werkzeug hash?
        if stored_hash_or_plain.startswith(("pbkdf2:", "scrypt:", "argon2:")) or stored_hash_or_plain.count("$") >= 2:
            return check_password_hash(stored_hash_or_plain, provided)
    except Exception:
        pass
    # Fallback: legacy plaintext compare
    return stored_hash_or_plain == provided


from sqlalchemy import or_

def _purge_user_relations(user_id: int):
    """
    Hard-delete all rows that reference the given user to allow a real user delete.
    Returns a dict of counts for messaging.
    """
    counts = {"predictions": 0, "appraisals": 0, "enrollments": 0, "perf_reviews": 0, "retention_plans": 0}

    # Predictions created by the user
    try:
        counts["predictions"] = Prediction.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    except Exception:
        db.session.rollback()

    # Appraisals where the user is employee OR supervisor
    try:
        counts["appraisals"] = (
            Appraisal.query
            .filter(or_(Appraisal.employee_id == user_id, Appraisal.supervisor_id == user_id))
            .delete(synchronize_session=False)
        )
    except Exception:
        db.session.rollback()

    # Cycle enrollments where user is employee OR manager
    try:
        counts["enrollments"] = (
            CycleEnrollment.query
            .filter(or_(CycleEnrollment.employee_id == user_id, CycleEnrollment.manager_id == user_id))
            .delete(synchronize_session=False)
        )
    except Exception:
        db.session.rollback()

    # Performance reviews (only if your model has these columns)
    try:
        cols = PerformanceReview.__table__.c.keys()
        conds = []
        if "employee_id" in cols:
            conds.append(PerformanceReview.employee_id == user_id)
        if "reviewer_id" in cols:
            conds.append(PerformanceReview.reviewer_id == user_id)
        if conds:
            counts["perf_reviews"] = PerformanceReview.query.filter(or_(*conds)).delete(synchronize_session=False)
    except Exception:
        db.session.rollback()

    # Retention plans (best-effort: check common columns)
    try:
        cols = RetentionPlan.__table__.c.keys()
        conds = []
        if "user_id" in cols:
            conds.append(RetentionPlan.user_id == user_id)
        if "employee_id" in cols:
            conds.append(RetentionPlan.employee_id == user_id)
        if "created_by" in cols:
            conds.append(RetentionPlan.created_by == user_id)
        if conds:
            counts["retention_plans"] = RetentionPlan.query.filter(or_(*conds)).delete(synchronize_session=False)
    except Exception:
        db.session.rollback()

    return counts


# ---------------- App / Config ----------------
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-12345')

db_url = os.getenv("DATABASE_URL", "sqlite:///app.db")

# Heroku/Railway sometimes give "postgres://"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Force SSL for managed Postgres (Railway usually requires it)
if db_url.startswith("postgresql://") and "sslmode" not in db_url:
    sep = "&" if "?" in db_url else "?"
    db_url = f"{db_url}{sep}sslmode=require"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Nice-to-haves for cloud DBs
app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {"pool_pre_ping": True})

csrf = CSRFProtect(app)

# Mail (optional)
if MAIL_AVAILABLE:
    app.config.setdefault("MAIL_SUPPRESS_SEND", False)
    app.config.setdefault("MAIL_DEFAULT_SENDER", os.getenv("MAIL_DEFAULT_SENDER", "noreply@example.com"))
    mail = Mail(app)
else:
    mail = None

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

account_bp = Blueprint('account', __name__, url_prefix='/account')

def safe_url_for(endpoint, **values):
    try: return url_for(endpoint, **values)
    except BuildError: return None
app.jinja_env.globals['safe_url_for'] = safe_url_for

def role_str(value):
    try: return (_role_to_string(value) or "").strip()
    except Exception: return str(value) if value is not None else ""
app.jinja_env.filters['role_str'] = role_str
app.jinja_env.globals['role_str'] = role_str

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

@app.context_processor
def inject_now():
    return {"current_year": datetime.utcnow().year}

# Init extensions / blueprints
db.init_app(app)
migrate.init_app(app, db)
app.register_blueprint(reviews_bp)
app.register_blueprint(appraisal_bp)

if os.getenv("AUTO_CREATE_DB", "1") == "1":
    with app.app_context():
        db.create_all()


# app.py (or your routes file)
from flask import request, redirect, url_for, flash
from werkzeug.security import generate_password_hash
from sqlalchemy import or_

ALLOWED_ROLES = {"EMPLOYEE", "MANAGER", "HR", "ADMIN"}

@app.post("/signup")
def signup():
    # Forward old /signup posts to the canonical /register handler
    return redirect(url_for("register"))



# ---------------- Admin bootstrap ----------------
def bootstrap_admin():
    try:
        email    = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "admin@local")
        username = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
        password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "Admin#12345")
        force    = os.getenv("BOOTSTRAP_RESET_ADMIN_PASSWORD", "0") == "1"

        u = (User.query.filter(func.lower(User.username) == username.lower()).first()
             or User.query.filter(func.lower(User.email) == email.lower()).first())

        if not u:
            u = User(username=username, email=email, role=Role.ADMIN, is_active=True)
            db.session.add(u)

        # Always enforce role/active
        u.role = Role.ADMIN
        u.is_active = True

        # Only set password when creating, when empty, or when explicitly forced
        if not u.password or force:
            u.password = generate_password_hash(password)

        db.session.commit()
        app.logger.warning("✅ Admin ensured: %s <%s>", username, u.email)
    except Exception as e:
        db.session.rollback()
        app.logger.error("bootstrap_admin failed: %s", e)





with app.app_context():
    bootstrap_admin()

# ---------------- Login Manager ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# add near your other HR exports
from flask import send_file
import io


from functools import wraps

def roles_required(*allowed_roles):
    """
    Usage: @roles_required("HR") or @roles_required("HR", "ADMIN")
    Ensures the current user has one of the allowed roles.
    """
    # Normalize the role strings once
    normalized = {(_role_to_string(r) or "").strip().upper() for r in allowed_roles}

    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapper(*args, **kwargs):
            role = get_role(current_user)
            if role in normalized:
                return fn(*args, **kwargs)
            flash("Access denied.", "danger")
            # Send them to the closest matching login modal
            target = (next(iter(normalized)) or "EMPLOYEE").lower()
            return redirect(url_for("home", open=f"login_{target}"))
        return wrapper
    return decorator



@app.route("/hr/export/pdf")
@login_required
@roles_required("HR")   # or whatever decorator you use
def hr_export_pdf():
    # TODO: replace with real PDF generation
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n% Fake minimal PDF ...\n")  # placeholder
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="hr_report.pdf",
        mimetype="application/pdf"
    )


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Model / Encoder loading (safe) ----------------
def _list_with_ids(pattern, prefix):
    out = {}
    for p in glob.glob(pattern):
        name = os.path.basename(p)
        ident = name.replace(prefix, "").rsplit(".", 1)[0]
        out[ident] = p
    return out

def load_paired_model_and_encoder():
    # Prefer explicit root files
    root_model = "model.pkl"; root_encoder = "encoder.pkl"
    if os.path.exists(root_model) and os.path.exists(root_encoder):
        try:
            m, e = joblib.load(root_model), joblib.load(root_encoder)
            print("Loaded root pair: model.pkl + encoder.pkl")
            return m, e
        except Exception:
            pass

    model_map = _list_with_ids("models/model_*.pkl", "model_")
    enc_map   = _list_with_ids("models/encoder_*.pkl", "encoder_")
    common_ids = set(model_map).intersection(enc_map)
    if common_ids:
        newest_id = max(common_ids, key=lambda i: os.path.getctime(model_map[i]))
        try:
            print(f"Loaded paired: {os.path.basename(model_map[newest_id])} + {os.path.basename(enc_map[newest_id])}")
            return joblib.load(model_map[newest_id]), joblib.load(enc_map[newest_id])
        except Exception:
            pass

    # nothing found -> return None to avoid crashes
    print("No model/encoder found. Predict feature disabled until training.")
    return None, None

model, encoder = load_paired_model_and_encoder()

# Expected columns (fallbacks)
CATEGORICAL_COLS = ['JobRole','Department','MaritalStatus','Gender','OverTime','BusinessTravel','EducationField']
NUMERIC_COLS = [
    'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate',
    'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked',
    'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel',
    'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
    'YearsSinceLastPromotion','YearsWithCurrManager'
]
try:
    # Try models/columns.joblib first, then root
    cols_path = "models/columns.joblib" if os.path.exists("models/columns.joblib") else "columns.joblib"
    if os.path.exists(cols_path):
        cols = joblib.load(cols_path)
        CATEGORICAL_COLS = list(cols.get("categorical_columns", CATEGORICAL_COLS))
        NUMERIC_COLS = list(cols.get("numerical_columns", NUMERIC_COLS))
        print("Loaded column order from", cols_path)
except Exception as e:
    print("Warning: couldn't load columns.joblib:", e)

def _build_encoder_input_df(form, encoder_obj):
    if not encoder_obj:
        return pd.DataFrame([{}]), []
    if hasattr(encoder_obj, "feature_names_in_"):
        enc_cols = list(encoder_obj.feature_names_in_)
    else:
        enc_cols = CATEGORICAL_COLS

    row = {}
    cats = getattr(encoder_obj, "categories_", None)
    for i, col in enumerate(enc_cols):
        v = form.get(col, "")
        if v != "":
            row[col] = v
        else:
            if cats is not None and i < len(cats) and len(cats[i]) > 0:
                row[col] = cats[i][0]
            else:
                row[col] = "0"

    df = pd.DataFrame([row], columns=enc_cols)
    for c in enc_cols:
        df[c] = df[c].astype(str)
    return df, enc_cols

def _risk_bucket(p: float):
    if p >= 0.70: return "High", "danger"
    if p >= 0.40: return "Moderate", "warning"
    return "Low", "success"

def _build_top_drivers(x_row, feature_names, coef, k=8):
    rows = []
    for i, name in enumerate(feature_names):
        val = float(x_row[i]); w = float(coef[i]); contrib = val * w
        rows.append({"name": name, "value": val, "weight": w,
                     "contribution": contrib, "direction": "↑ risk" if contrib > 0 else "↓ risk"})
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows[:k]

def _recommendations(payload: dict, top_drivers: list, will_leave: bool):
    rec = []
    top_names = ", ".join(d["name"] for d in top_drivers[:3]) or "multiple factors"
    rec.append(f"Focus on: {top_names}.")
    try:
        js = int(payload.get("JobSatisfaction", 3))
        es = int(payload.get("EnvironmentSatisfaction", 3))
        wlb = int(payload.get("WorkLifeBalance", 3))
        dist = int(payload.get("DistanceFromHome", 0))
        yslp = int(payload.get("YearsSinceLastPromotion", 0))
        train = int(payload.get("TrainingTimesLastYear", 0))
        income = float(payload.get("MonthlyIncome", 0.0))
    except Exception:
        js = es = wlb = 3; dist = yslp = train = 0; income = 0.0
    if will_leave:
        if payload.get("OverTime") == "Yes": rec.append("Reduce sustained overtime; offer comp time or flexible scheduling.")
        if js <= 2: rec.append("Run a 1:1 to identify role fit; agree a 30-day plan.")
        if es <= 2: rec.append("Fix workspace/environment issues raised in feedback.")
        if wlb <= 2: rec.append("Improve work–life balance; encourage PTO.")
        if dist >= 20: rec.append("Consider hybrid/remote days.")
        if yslp >= 3: rec.append("Discuss growth; schedule a promotion review.")
        if train == 0: rec.append("Fund upskilling this quarter.")
        if income and income < 4000: rec.append("Benchmark comp vs band; adjust if below peers.")
    else:
        rec.append("Maintain recognition and quarterly growth check-ins.")
        rec.append("Keep workload steady; avoid sustained overtime.")
    return rec[:8]

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template("home.html")

# ---------- Auth ----------
from urllib.parse import urlparse, urljoin
def _is_safe_next(target: str) -> bool:
    if not target: return False
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return (test.scheme in ("http", "https")) and (ref.netloc == test.netloc)

@app.route("/login", methods=["GET", "POST"])
@csrf.exempt
def login():
    # We don’t render a separate login page; open the modal on home instead.
    if request.method == "GET":
        return redirect(url_for("home", open="login"))

    identifier = (request.form.get("identifier") or "").strip()
    password   = (request.form.get("password")   or "").strip()
    expected   = (request.form.get("expected_role") or "").strip().upper() or None

    # Case-insensitive lookup
    user = User.query.filter(
        or_(
            func.lower(User.username) == identifier.lower(),
            func.lower(User.email) == identifier.lower()
        )
    ).first()

    open_target = f"login_{expected.lower()}" if expected else "login"
    # UPDATED: tolerant password verification (hash or legacy plaintext)
    stored = getattr(user, "password", "") if user else ""
    if (not user) or (not _verify_password(stored or "", password)):
        flash("Invalid username/email or password.", "danger")
        return redirect(url_for("home", open=open_target))

    if not getattr(user, "is_active", True):
        flash("Your account is pending Admin approval.", "warning")
        return redirect(url_for("home", open=open_target))

    if expected and get_role(user) != expected:
        flash(f"This portal is for {expected.title()} accounts.", "danger")
        return redirect(url_for("home", open=open_target))

    login_user(user)
    session["user_id"]  = user.id
    session["username"] = user.username
    session["role"]     = get_role(user)

    nxt = request.args.get("next")
    if _is_safe_next(nxt):
        return redirect(nxt)
    return redirect(url_for("dashboard"))

# ----- Register (Admin-created accounts auto-activate) -----
@app.route('/register', methods=['POST'])
@csrf.exempt
def register():
    def clean(v):
        if v is None: return None
        v = v.strip()
        return v or None

    username = clean(request.form.get('username'))
    email    = clean(request.form.get('email'))
    role_str = (request.form.get('role') or '').strip().upper()
    raw_pw   = request.form.get('password') or ''
    confirm  = request.form.get('confirm_password') or ''

    if not username or not email or not role_str:
        flash('Please fill all required fields.', 'danger')
        return redirect(url_for('home', open='signup'))
    if len(raw_pw) < 12:
        flash('Password must be at least 12 characters.', 'danger')
        return redirect(url_for('home', open='signup'))

    if raw_pw != confirm:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('home', open='signup'))

    try:
        role_val = Role(role_str)
    except Exception:
        flash('Invalid role selected.', 'danger')
        return redirect(url_for('home', open='signup'))

    hashed_password = generate_password_hash(raw_pw)

    if role_val == Role.EMPLOYEE:
        full_name    = clean(request.form.get('name'))
        employee_id  = clean(request.form.get('employee_id'))
        department   = clean(request.form.get('department'))
        job_title    = clean(request.form.get('title'))
        manager_name = clean(request.form.get('manager_name'))
        phone        = clean(request.form.get('phone'))
        location     = clean(request.form.get('location'))
    else:
        full_name = employee_id = department = job_title = manager_name = phone = location = None

    if User.query.filter(func.lower(User.username) == username.lower()).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('home', open='signup'))
    if User.query.filter(func.lower(User.email) == email.lower()).first():
        flash('Email already exists.', 'danger')
        return redirect(url_for('home', open='signup'))
    if employee_id and User.query.filter_by(employee_id=employee_id).first():
        flash('Employee ID already exists.', 'danger')
        return redirect(url_for('home', open='signup'))

    auto_active = current_user.is_authenticated and get_role(current_user) == "ADMIN"

    user = User(
        username=username,
        email=email,
        password=hashed_password,
        full_name=full_name,
        employee_id=employee_id,
        department=department,
        job_title=job_title,
        manager_name=manager_name,
        phone=phone,
        location=location,
        role=role_val,
        is_active=bool(auto_active),
        manager_id=None
    )

    db.session.add(user)
    try:
        db.session.commit()
        flash('✅ Account created and activated.' if auto_active else 'Account created. Awaiting admin approval.', 'success')
        return redirect(url_for('home', open=f"login_{role_val.value.lower()}"))
    except IntegrityError as e:
        db.session.rollback()
        msg_l = str(e.orig)
        if 'users.username' in msg_l: msg = 'Username already exists.'
        elif 'users.email' in msg_l: msg = 'Email already exists.'
        elif 'users.employee_id' in msg_l: msg = 'Employee ID already exists. Leave it blank unless registering an Employee.'
        else: msg = 'Registration failed.'
        flash(msg, 'danger')
        return redirect(url_for('home', open='signup'))


@app.teardown_request
def _teardown_request(exc):
    # If the request raised, roll back so the connection isn't left aborted
    if exc is not None:
        try:
            db.session.rollback()
        except Exception:
            pass
    # Always remove the scoped session at the end of the request
    try:
        db.session.remove()
    except Exception:
        pass


# ----- Account/Profile -----
@account_bp.post('/profile/update')
@login_required
def update_profile():
    f = request.form
    nm   = (f.get('name') or f.get('full_name') or current_user.full_name)
    dept = (f.get('department') or current_user.department)
    ttl  = (f.get('title') or f.get('job_title') or current_user.job_title)
    mng  = (f.get('manager_name') or current_user.manager_name)
    ph   = (f.get('phone') or current_user.phone)
    loc  = (f.get('location') or current_user.location)
    eid  = (f.get('employee_id') or '').strip()

    current_user.full_name    = nm
    current_user.department   = dept
    current_user.job_title    = ttl
    current_user.manager_name = mng
    current_user.phone        = ph
    current_user.location     = loc
    if eid:
        current_user.employee_id = eid

    pw  = f.get('password') or ''
    pw2 = f.get('password_confirm') or ''
    if pw or pw2:
        if pw != pw2:
            msg = 'Passwords do not match.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(ok=False, message=msg), 400
            flash(msg, 'danger'); return redirect(request.referrer or url_for('dashboard'))
        if len(pw) < 8:
            msg = 'Password must be at least 8 characters.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(ok=False, message=msg), 400
            flash(msg, 'danger'); return redirect(request.referrer or url_for('dashboard'))
        current_user.password = generate_password_hash(pw)

    db.session.commit()

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(
            ok=True, message='Profile updated.',
            user={
                "name": current_user.full_name,
                "department": current_user.department,
                "title": current_user.job_title,
                "manager_name": current_user.manager_name,
                "phone": current_user.phone,
                "location": current_user.location,
                "employee_id": current_user.employee_id,
                "role": get_role(current_user),
            }
        )
    flash('Profile updated.', 'success')
    return redirect(request.referrer or url_for('dashboard'))

app.register_blueprint(account_bp)

# ----- Password recovery -----
@app.route('/recover', methods=['GET', 'POST'])
@csrf.exempt
def recover():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        user = User.query.filter(func.lower(User.email) == email.lower()).first()
        if user and MAIL_AVAILABLE and mail:
            try:
                token = serializer.dumps(email, salt='password-recover')
                reset_url = url_for('reset_password', token=token, _external=True)
                msg = Message("Password Reset Request", recipients=[email])
                msg.body = f"Click the link to reset your password:\n{reset_url}\n\nThis link is valid for 1 hour."
                mail.send(msg)
                flash("Password reset link sent to your email.", "info")
                return redirect(url_for('home', open='login'))
            except Exception as e:
                app.logger.warning("Mail send failed: %s", e)
                flash("Email service is not configured.", "warning")
        flash("If the email exists, a reset link will be sent.", "info")
        return redirect(url_for('home', open='login'))

    # Try to render template if present; else serve a tiny fallback page.
    try:
        return render_template('recover.html')
    except Exception:
        return render_template_string("""
        <!doctype html><title>Recover</title>
        <div class="container" style="max-width:560px;margin:40px auto;font-family:system-ui">
          <h3>Password recovery</h3>
          <form method="post">
            <label>Email</label><br>
            <input name="email" type="email" required style="width:100%;padding:8px;margin:8px 0">
            <button type="submit" style="padding:8px 12px">Send reset link</button>
          </form>
          <p><a href="{{ url_for('home') }}">← Back to Home</a></p>
        </div>
        """)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
@csrf.exempt
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-recover', max_age=3600)
    except SignatureExpired:
        flash("The reset link has expired.", "danger")
        return redirect(url_for('recover'))
    except BadSignature:
        flash("Invalid reset token.", "danger")
        return redirect(url_for('recover'))

    if request.method == 'POST':
        new_password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(request.url)
        user = User.query.filter(func.lower(User.email) == email.lower()).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("Your password has been updated. Please log in.", "success")
            return redirect(url_for('home', open='login'))

    try:
        return render_template('reset_password.html', token=token)
    except Exception:
        return render_template_string("""
        <!doctype html><title>Reset password</title>
        <div class="container" style="max-width:560px;margin:40px auto;font-family:system-ui">
          <h3>Set a new password</h3>
          <form method="post">
            <input name="password" type="password" placeholder="New password" required style="width:100%;padding:8px;margin:8px 0">
            <input name="confirm_password" type="password" placeholder="Confirm password" required style="width:100%;padding:8px;margin:8px 0">
            <button type="submit" style="padding:8px 12px">Update</button>
          </form>
        </div>
        """)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home', open='login'))

# ----- Dashboards -----
@app.route("/dashboard")
@login_required
def dashboard():
    role = _current_role()
    if role == "EMPLOYEE": return redirect(url_for("dashboard_employee"))
    if role == "MANAGER":  return redirect(url_for("dashboard_manager"))
    if role == "HR":       return redirect(url_for("dashboard_hr"))
    if role == "ADMIN":    return redirect(url_for("dashboard_admin"))
    return render_template("dashboard_generic.html")

@app.route("/dashboard/employee")
@login_required
def dashboard_employee():
    role = _current_role()
    if role in ("MANAGER", "HR", "ADMIN"):
        return redirect(url_for(f"dashboard_{role.lower()}"))

    try:
        db.session.rollback()
    except Exception:
        pass

    ctx = dict(
        active_appraisal=None, self_progress=0, section_progress=[],
        goals=[], achievements=[], feedback=[], announcements=[],
        appraisals_count=0, current_cycle=None, appraisal_deadline=None,
    )

    try:
        current_cycle = globals().get("get_current_cycle", lambda: None)()
        ctx["current_cycle"] = current_cycle

        q = (Appraisal.query
             .options(joinedload(Appraisal.workload), joinedload(Appraisal.scores))
             .filter_by(employee_id=current_user.id))
        if current_cycle:
            q = q.filter_by(cycle=current_cycle)
        active = q.order_by(Appraisal.created_at.desc()).first()
        ctx["active_appraisal"] = active
        if active:
            ctx["self_progress"] = getattr(active, "progress", 0)
            ctx["section_progress"] = getattr(active, "section_progress", [])
            if getattr(active, "deadline", None):
                ctx["appraisal_deadline"] = active.deadline

        try:
            ctx["appraisals_count"] = Appraisal.query.filter(
                Appraisal.employee_id == current_user.id,
                or_(
                    Appraisal.status == "completed",
                    Appraisal.status == ReviewStatus.APPROVED,
                    Appraisal.status == ReviewStatus.HR_REVIEWED
                )
            ).count()
        except Exception:
            ctx["appraisals_count"] = 0
    except Exception as e:
        db.session.rollback()
        app.logger.warning("dashboard_employee build failed: %s", e)

    return render_template("dashboard_employee.html", **ctx)

@app.route("/history")
@login_required
def prediction_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=predictions)

@app.route('/dashboard/manager')
@login_required
def dashboard_manager():
    if get_role(current_user) != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_manager"))

    mid = current_user.id
    def _count_by_status(*statuses):
        q = (db.session.query(func.count(func.distinct(Appraisal.id)))
             .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
             .filter(or_(Appraisal.supervisor_id == mid, CycleEnrollment.manager_id == mid)))
        if len(statuses) == 1:
            q = q.filter(Appraisal.status == statuses[0])
        else:
            q = q.filter(Appraisal.status.in_(statuses))
        return q.scalar() or 0

    pending_count     = _count_by_status(ReviewStatus.SUBMITTED_TO_MANAGER)
    awaiting_hr_count = _count_by_status(ReviewStatus.MANAGER_SUBMITTED)
    returned_count    = _count_by_status(ReviewStatus.REJECTED)
    completed_count   = _count_by_status(ReviewStatus.HR_REVIEWED, ReviewStatus.APPROVED)

    team_members = (db.session.query(func.count(func.distinct(User.id)))
                    .join(CycleEnrollment, CycleEnrollment.employee_id == User.id)
                    .filter(CycleEnrollment.manager_id == mid, User.is_active.is_(True))
                    .scalar()) or 0

    stats = dict(
        team_members=team_members, pending=pending_count, awaiting_hr=awaiting_hr_count,
        returned=returned_count, completed=completed_count
    )
    ctx = {
        "pending_count": pending_count, "awaiting_hr_count": awaiting_hr_count,
        "returned_count": returned_count, "completed_count": completed_count,
        "team_members": team_members, "stats": stats,
        # aliases for templates
        "pending": pending_count, "pending_total": pending_count,
        "awaiting_hr": awaiting_hr_count, "submitted_count": awaiting_hr_count,
        "completed": completed_count, "completed_total": completed_count,
        "team_count": team_members, "members_count": team_members,
        "cards": stats, "metrics": stats,
    }
    return render_template("manager_dashboard.html", **ctx)

@app.route('/dashboard/hr')
@login_required
def dashboard_hr():
    if get_role(current_user) != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_hr"))

    def sfloat(x):
        try: return float(x)
        except: return None

    def as_dict(payload):
        if isinstance(payload, dict): return payload
        try: return json.loads(payload or "{}")
        except Exception: return {}

    try:
        total_employees = User.query.filter_by(role=Role.EMPLOYEE, is_active=True).count()
    except Exception:
        total_employees = User.query.filter_by(role=Role.EMPLOYEE).count()
    try:
        total_appraisals = PerformanceReview.query.count()
    except Exception:
        total_appraisals = 0

    status_dist = {}
    try:
        for st in ReviewStatus:
            status_dist[st.name] = PerformanceReview.query.filter_by(status=st).count()
    except Exception:
        status_dist = {}

    try:
        preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5000).all()
    except Exception:
        preds = []

    yes_count = 0
    risk_over_time, dept_stats = {}, {}
    ot_stats = {"Yes": {"yes": 0, "tot": 0}, "No": {"yes": 0, "tot": 0}}
    tenure_bins = {"<1y": {"yes": 0, "tot": 0},"1–3y": {"yes": 0, "tot": 0},"3–5y": {"yes": 0, "tot": 0},
                   "5–10y": {"yes": 0, "tot": 0},"10y+": {"yes": 0, "tot": 0}}
    recent_rows = []

    for p in preds:
        ts = getattr(p, "timestamp", None)
        proba = sfloat(getattr(p, "confidence", None))
        result_text = (getattr(p, "result", "") or "").strip()
        data = as_dict(getattr(p, "input_data", {}))
        recent_rows.append({
            "ts": ts.strftime("%Y-%m-%d %H:%M") if ts else "",
            "result": result_text, "proba": f"{(proba or 0)*100:.1f}%",
            "dept": data.get("Department", ""), "role": data.get("JobRole", ""),
            "ot": data.get("OverTime", ""), "years": data.get("YearsAtCompany", ""),
        })
        if result_text.lower() == "yes": yes_count += 1
        if ts and proba is not None:
            risk_over_time.setdefault(ts.date().isoformat(), []).append(proba)
        dept = data.get("Department", "Unknown")
        dept_stats.setdefault(dept, {"yes": 0, "tot": 0})
        dept_stats[dept]["tot"] += 1
        if result_text.lower() == "yes": dept_stats[dept]["yes"] += 1
        ot = str(data.get("OverTime", "Unknown"))
        if ot in ot_stats:
            ot_stats[ot]["tot"] += 1
            if result_text.lower() == "yes": ot_stats[ot]["yes"] += 1
        try:
            y = float(data.get("YearsAtCompany", 0))
        except Exception:
            y = 0
        if y < 1: b = "<1y"
        elif y < 3: b = "1–3y"
        elif y < 5: b = "3–5y"
        elif y < 10: b = "5–10y"
        else: b = "10y+"
        tenure_bins[b]["tot"] += 1
        if result_text.lower() == "yes": tenure_bins[b]["yes"] += 1

    total_predictions = len(preds)
    yes_rate = round(100.0 * yes_count / total_predictions, 1) if total_predictions else 0.0
    trend_dates = sorted(risk_over_time.keys())
    trend_avg = [round(100.0 * (sum(risk_over_time[d]) / max(len(risk_over_time[d]), 1)), 2) for d in trend_dates]

    dept_labels, dept_rates = [], []
    for d, obj in sorted(dept_stats.items(), key=lambda kv: kv[0]):
        rate = 100.0 * obj["yes"] / obj["tot"] if obj["tot"] else 0.0
        dept_labels.append(d); dept_rates.append(round(rate, 2))
    ot_labels = list(ot_stats.keys())
    ot_rates = [round(100.0 * v["yes"] / v["tot"], 2) if v["tot"] else 0.0 for v in ot_stats.values()]
    tenure_labels = list(tenure_bins.keys())
    tenure_rates = [round(100.0 * v["yes"] / v["tot"], 2) if v["tot"] else 0.0 for v in tenure_bins.values()]

    awaiting_rows = (Appraisal.query
                     .filter(Appraisal.status == ReviewStatus.MANAGER_SUBMITTED)
                     .order_by(Appraisal.created_at.desc())
                     .all())
    hr_awaiting_count = len(awaiting_rows)
    hr_returned_count = (db.session.query(func.count(Appraisal.id))
                         .filter(Appraisal.status == ReviewStatus.REJECTED).scalar() or 0)
    hr_finalized_count = (db.session.query(func.count(Appraisal.id))
                          .filter(Appraisal.status == ReviewStatus.APPROVED).scalar() or 0)
    hr_overdue_count = (db.session.query(func.count(Appraisal.id))
                        .filter(Appraisal.period_end != None,
                                Appraisal.period_end < date.today(),
                                Appraisal.status != ReviewStatus.APPROVED).scalar() or 0)

    status_pairs = (db.session.query(Appraisal.status, func.count(Appraisal.id))
                    .group_by(Appraisal.status).all())
    status_labels = [getattr(s, "value", str(s)) for s, _ in status_pairs]
    status_counts = [c for _, c in status_pairs]

    emp_ids = {a.employee_id for a in awaiting_rows}
    mgr_ids = {a.supervisor_id for a in awaiting_rows}
    order_col = getattr(Appraisal, "updated_at", getattr(Appraisal, "created_at", Appraisal.id))
    recent_approved = (Appraisal.query
                       .filter(Appraisal.status == ReviewStatus.APPROVED)
                       .order_by(order_col.desc()).limit(12).all())
    emp_ids |= {a.employee_id for a in recent_approved}
    mgr_ids |= {a.supervisor_id for a in recent_approved}

    ids = list(emp_ids | mgr_ids)
    user_map = {u.id: u for u in (User.query.filter(User.id.in_(ids)).all() if ids else [])}

    hr_queue = []
    for a in awaiting_rows:
        hr_queue.append({
            "id": a.id, "employee_id": a.employee_id,
            "employee_name": getattr(user_map.get(a.employee_id), "username", None),
            "manager_id": a.supervisor_id,
            "manager_name": getattr(user_map.get(a.supervisor_id), "username", None),
            "period_start": a.period_start, "period_end": a.period_end,
            "total_score": a.total_score,
            "status": getattr(a.status, "value", str(a.status)),
            "review_url": url_for("appraisal.hr_review_queue_ui"),
            "return_url": url_for("appraisal.hr_review_queue_ui"),
            "finalize_url": url_for("appraisal.hr_review_queue_ui"),
        })

    recent_appraisals = [{
        "id": a.id, "employee_id": a.employee_id,
        "employee_name": getattr(user_map.get(a.employee_id), "username", None),
        "period_start": a.period_start, "period_end": a.period_end,
        "total_score": a.total_score,
        "finalized_at": getattr(a, "updated_at", None),
    } for a in recent_approved]

    selected_cycle_id = request.args.get("cycle_id")

    try:
        total_appraisals_real = db.session.query(func.count(Appraisal.id)).scalar() or 0
    except Exception:
        total_appraisals_real = 0

    return render_template(
        "hr_dashboard.html",
        total_employees=total_employees or 0,
        total_appraisals=total_appraisals_real,
        appraisals_total=total_appraisals_real,
        total_predictions=len(preds) or 0,
        yes_rate=yes_rate or 0.0,
        avg_conf_yes=0.0,
        trend_dates=trend_dates or [], trend_avg=trend_avg or [],
        dept_labels=dept_labels or [], dept_rates=dept_rates or [],
        ot_labels=ot_labels or [], ot_rates=ot_rates or [],
        tenure_labels=tenure_labels or [], tenure_rates=tenure_rates or [],
        recent_rows=recent_rows[:12] or [],
        hr_awaiting_count=hr_awaiting_count, hr_returned_count=hr_returned_count,
        hr_finalized_count=hr_finalized_count, hr_overdue_count=hr_overdue_count,
        hr_queue=hr_queue, status_labels=status_labels or [], status_counts=status_counts or [],
        recent_appraisals=recent_appraisals or [], selected_cycle_id=selected_cycle_id
    )

# ----- Exports -----
@app.route("/hr/export/csv")
@login_required
def hr_export_csv():
    if get_role(current_user) != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard_hr"))

    preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    rows = []
    for p in preds:
        data = p.input_data if isinstance(p.input_data, dict) else json.loads(p.input_data or "{}")
        row = {"timestamp": getattr(p, "timestamp", None), "result": getattr(p, "result", ""),
               "confidence": getattr(p, "confidence", "")}
        row.update(data or {})
        rows.append(row)

    df = pd.DataFrame(rows)
    buf = BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"hr_predictions_{datetime.utcnow():%Y%m%d}.csv",
                     mimetype="text/csv")

@app.route("/hr/export/excel")
@login_required
def hr_export_excel():
    if get_role(current_user) != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard_hr"))

    preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    rows = []
    for p in preds:
        data = p.input_data if isinstance(p.input_data, dict) else json.loads(p.input_data or "{}")
        row = {"timestamp": getattr(p, "timestamp", None), "result": getattr(p, "result", ""),
               "confidence": getattr(p, "confidence", "")}
        row.update(data or {})
        rows.append(row)

    df = pd.DataFrame(rows)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Predictions")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"hr_predictions_{datetime.utcnow():%Y%m%d}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----- Notifications for templates -----
@app.context_processor
def inject_notifications():
    count = 0; items = []
    if current_user.is_authenticated:
        try:
            if get_role(current_user) in ["MANAGER", "HR"]:
                count = Prediction.query.filter_by(user_id=current_user.id).count()
                items = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()
            if get_role(current_user) == "EMPLOYEE":
                count = Appraisal.query.filter_by(employee_id=current_user.id, status=ReviewStatus.DRAFT).count()
        except Exception as e:
            app.logger.warning("inject_notifications failed: %s", e)
            db.session.rollback()
    return dict(notif_count=count, notif_items=items)


# ----- Predict -----
# ----- Predict -----
@app.route('/predict', methods=['GET', 'POST'])
@login_required
@csrf.exempt
def predict():
    # GET -> show the form
    if request.method == 'GET':
        try:
            return render_template('predict_form.html')
        except Exception:
            return render_template_string(
                "<p style='padding:20px;font-family:system-ui'>Upload/train a model first.</p>"
            )

    # POST -> run prediction
    # treat as ajax only if explicitly asked for JSON
    wants_json = (request.headers.get('X-Requested-With') == 'XMLHttpRequest')

    # Force HTML for classic form posts (prevents accidental JSON if a header sneaks in)
    is_form_post = request.mimetype in ("application/x-www-form-urlencoded", "multipart/form-data", None)
    if is_form_post:
        wants_json = False

    if model is None or encoder is None:
        if wants_json:
            return jsonify(
                ok=False,
                error="MODEL_UNAVAILABLE",
                message="Prediction model not available yet. Ask an admin to (re)train it."
            ), 503
        flash("Prediction model not available yet. Train a model from the Admin dashboard.", "warning")
        return redirect(url_for('predict'))

    try:
        form = request.form

        # 1) Numeric inputs: blank -> 0.0, only flag non-numeric text
        num_vals, bad = {}, []
        for col in NUMERIC_COLS:
            raw = (form.get(col, "") or "").strip()
            if raw == "":
                num_vals[col] = 0.0
            else:
                try:
                    num_vals[col] = float(raw)
                except ValueError:
                    bad.append(col)
        if bad:
            if wants_json:
                return jsonify(
                    ok=False,
                    error="BAD_NUMERIC_FIELDS",
                    fields=bad,
                    message=f"These numeric fields are invalid: {', '.join(bad)}"
                ), 400
            flash(f"These numeric fields are invalid: {', '.join(bad)}", 'danger')
            return redirect(url_for('predict'))

        x_num = np.array([num_vals[col] for col in NUMERIC_COLS], dtype=float).reshape(1, -1)

        # 2) Categorical inputs: build encoder frame with safe defaults
        enc_df, enc_cols = _build_encoder_input_df(form, encoder)
        x_enc = encoder.transform(enc_df)

        # 3) Match model input expectations
        n_model_in = getattr(model, 'n_features_in_', None)
        if n_model_in is not None and x_enc.shape[1] == n_model_in:
            x_final = x_enc
            feature_names_prefix = []
        elif n_model_in is not None and (x_enc.shape[1] + x_num.shape[1]) == n_model_in:
            x_final = np.hstack([x_num, x_enc])
            feature_names_prefix = NUMERIC_COLS
        else:
            x_final = np.hstack([x_num, x_enc])
            feature_names_prefix = NUMERIC_COLS

        # 4) Predict
        proba = float(model.predict_proba(x_final)[0][1])
        pred = proba >= 0.5

        # 5) Persist record
        payload = {k: form.get(k, "") for k in (NUMERIC_COLS + enc_cols)}
        rec = Prediction(
            user_id=current_user.id,
            input_data=payload,
            result=('Yes' if pred else 'No'),
            confidence=f"{proba:.3f}"
        )
        db.session.add(rec)
        db.session.commit()

        # 6) Build explanations (top drivers)
        try:
            feature_names_enc = list(encoder.get_feature_names_out(enc_cols))
        except Exception:
            feature_names_enc = list(getattr(encoder, "get_feature_names", lambda x: x)(enc_cols))

        feature_names = (feature_names_prefix + feature_names_enc)[:x_final.shape[1]]

        # Get coefficients safely (not all models expose coef_)
        if hasattr(model, "coef_") and model.coef_ is not None:
            coef_vec = np.asarray(model.coef_).reshape(-1)
        else:
            coef_vec = np.zeros(len(feature_names), dtype=float)

        coef = coef_vec[:len(feature_names)]
        x_row = x_final[0]
        top_drivers = _build_top_drivers(x_row, feature_names, coef, k=8)

        # 7) UX context
        risk_label, risk_color = _risk_bucket(proba)
        recs = _recommendations(payload, top_drivers, bool(pred))

        result_ctx = {
            "prediction_text": 'Employee will Attrit' if pred else 'Employee will NOT Attrit',
            "probability": f"{proba*100:.1f}%",
            "pred": 'Yes' if pred else 'No',
            "proba": proba,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "top_drivers": top_drivers,
            "recommendations": recs,
            "input_data": payload, "InputData": payload,
            "Result": 'Yes' if pred else 'No',
            "Confidence": f"{proba:.3f}",
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # JSON only when explicitly requested; otherwise render HTML
        if wants_json:
            return jsonify(ok=True, **result_ctx), 200

        try:
            return render_template('result.html', **result_ctx, results=[result_ctx])
        except Exception:
            return render_template_string("<pre>{{ctx|tojson(indent=2)}}</pre>", ctx=result_ctx)

    except Exception as e:
        db.session.rollback()
        if wants_json:
            return jsonify(ok=False, error="SERVER_ERROR", message=f"Prediction error: {e}"), 500
        flash(f"Prediction error: {e}", 'danger')
        return redirect(url_for('predict'))

@app.post("/retention/save")
@login_required
def save_retention_plan():
    # Minimal placeholder to avoid 404s and show a flash
    immediate = (request.form.get("immediate_plan") or "").strip()
    assigned  = (request.form.get("assigned_to") or "").strip()
    # TODO: persist to your RetentionPlan model if/when ready
    flash("Retention plan saved (demo).", "success")
    return redirect(url_for("predict"))


@app.route("/predict_form")
@login_required
def predict_form():
    return redirect(url_for("predict"))

# ----- Admin tools -----
@app.route('/admin/train_model', methods=['POST'])
@login_required
def train_model():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    try:
        dataset_path = os.path.join(app.root_path, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        if not os.path.exists(dataset_path):
            flash("Dataset not found for training.", "danger")
            return redirect(url_for('dashboard_admin'))

        df = pd.read_csv(dataset_path)
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

        X = df.drop(['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
        y = df['Attrition']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        categorical_columns = ['JobRole','Department','MaritalStatus','Gender','OverTime','BusinessTravel','EducationField']
        numerical_columns = [c for c in X.columns if c not in categorical_columns]

        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        enc.fit(X_train[categorical_columns])

        X_train_enc = enc.transform(X_train[categorical_columns])
        X_test_enc  = enc.transform(X_test[categorical_columns])

        X_train_final = np.hstack((X_train[numerical_columns].values, X_train_enc))
        X_test_final  = np.hstack((X_test[numerical_columns].values, X_test_enc))

        mdl = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000, C=0.5, random_state=42)
        mdl.fit(X_train_final, y_train)

        os.makedirs("models", exist_ok=True)
        ts = int(datetime.utcnow().timestamp())
        joblib.dump(mdl, f"models/model_{ts}.pkl")
        joblib.dump(enc, f"models/encoder_{ts}.pkl")
        joblib.dump({"categorical_columns": categorical_columns, "numerical_columns": numerical_columns},
                    "models/columns.joblib")

        flash("✅ Model retrained and saved successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"⚠️ Training failed: {e}", "danger")

    # reload pointers so predict works immediately
    global model, encoder
    model, encoder = load_paired_model_and_encoder()

    return redirect(url_for('dashboard_admin'))

@app.route('/dashboard/admin')
@login_required
def dashboard_admin():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    users = User.query.all()
    model_files = glob.glob("models/model_*.pkl")
    encoder_files = glob.glob("models/encoder_*.pkl")
    criteria = Criteria.query.all()

    models = [{"name": os.path.basename(f), "ctime": os.path.getctime(f)} for f in model_files]
    encoders = [{"name": os.path.basename(f), "ctime": os.path.getctime(f)} for f in encoder_files]

    return render_template("admin_dashboard.html", users=users, models=models, encoders=encoders, criteria=criteria)

@app.route('/admin/clear_predictions', methods=['POST'])
@login_required
def clear_predictions():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    try:
        Prediction.query.delete()
        db.session.commit()
        flash("All predictions have been cleared.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error clearing predictions: {e}", "danger")
    return redirect(url_for('dashboard_admin'))

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    try:
        pred = Prediction.query.get_or_404(prediction_id)
        if pred.user_id != current_user.id and get_role(current_user) not in ["ADMIN", "HR"]:
            flash("You don’t have permission to delete this prediction.", "danger")
            return redirect(url_for('prediction_history'))
        db.session.delete(pred); db.session.commit()
        flash("Prediction deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting prediction: {e}", "danger")
    return redirect(url_for('prediction_history'))

@app.route('/admin/create_user', methods=['POST'])
@login_required
def create_user():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    username = request.form.get('username'); email = request.form.get('email')
    password = request.form.get('password'); role = request.form.get('role')

    if not username or not email or not password or not role:
        flash("All fields are required", "danger"); return redirect(url_for('dashboard_admin'))
    if User.query.filter(func.lower(User.username) == username.lower()).first():
        flash("Username already exists", "danger"); return redirect(url_for('dashboard_admin'))
    if User.query.filter(func.lower(User.email) == email.lower()).first():
        flash("Email already registered", "danger"); return redirect(url_for('dashboard_admin'))

    new_user = User(username=username, email=email,
                    password=generate_password_hash(password),
                    role=Role(role), is_active=True)
    db.session.add(new_user); db.session.commit()
    flash(f"{role.title()} account '{username}' created successfully!", "success")
    return redirect(url_for('dashboard_admin'))

@app.route('/admin/approve/<int:user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))
    user = User.query.get_or_404(user_id); user.is_active = True; db.session.commit()
    flash(f"✅ User {user.username} has been approved.", "success")
    return redirect(url_for('dashboard_admin'))

@app.route('/admin/reject/<int:user_id>', methods=['POST'])
@login_required
def reject_user(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    user = User.query.get_or_404(user_id)

    try:
        # Purge all related data first
        counts = _purge_user_relations(user_id)

        # Now remove the user itself
        db.session.delete(user)
        db.session.commit()

        msg = (
            f"❌ User '{user.username}' rejected and permanently removed. "
            f"Deleted: predictions {counts['predictions']}, appraisals {counts['appraisals']}, "
            f"enrollments {counts['enrollments']}, performance reviews {counts['perf_reviews']}, "
            f"retention plans {counts['retention_plans']}."
        )
        flash(msg, "danger")
    except IntegrityError as e:
        db.session.rollback()
        flash(f"Could not fully remove user due to DB constraints: {e}", "danger")
    except Exception as e:
        db.session.rollback()
        flash(f"Error while removing user: {e}", "danger")

    return redirect(url_for('dashboard_admin'))


@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@login_required
def reset_user_password(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))
    user = User.query.get_or_404(user_id); new_pw = "changeme123"
    user.password = generate_password_hash(new_pw); db.session.commit()
    flash(f"🔑 Password for {user.username} has been reset to '{new_pw}'.", "warning")
    return redirect(url_for('dashboard_admin'))

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    u = User.query.get_or_404(user_id)

    try:
        # Purge all related data first
        counts = _purge_user_relations(user_id)

        # Now remove the user
        db.session.delete(u)
        db.session.commit()

        msg = (
            f"🗑️ User '{getattr(u, 'username', user_id)}' permanently removed. "
            f"Deleted: predictions {counts['predictions']}, appraisals {counts['appraisals']}, "
            f"enrollments {counts['enrollments']}, performance reviews {counts['perf_reviews']}, "
            f"retention plans {counts['retention_plans']}."
        )
        flash(msg, "success")
    except IntegrityError as e:
        db.session.rollback()
        flash(f"Could not fully remove user due to DB constraints: {e}", "danger")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {e}", "danger")

    return redirect(request.referrer or url_for("dashboard_admin"))


@app.route('/admin/delete_old_models', methods=['POST'])
@login_required
def delete_old_models():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))
    try:
        model_files = glob.glob("models/model_*.pkl")
        encoder_files = glob.glob("models/encoder_*.pkl")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            for f in model_files:
                if f != latest_model: os.remove(f)
        if encoder_files:
            latest_encoder = max(encoder_files, key=os.path.getctime)
            for f in encoder_files:
                if f != latest_encoder: os.remove(f)
        flash("🧹 Old models/encoders deleted. Kept latest.", "success")
    except Exception as e:
        flash(f"⚠️ Error deleting old models: {e}", "danger")
    return redirect(url_for('dashboard_admin'))

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)

