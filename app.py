from flask import Flask, render_template, request, session, redirect, url_for, flash, send_file, abort, jsonify, Blueprint, has_request_context, g
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from sqlalchemy import or_, func
from sqlalchemy.orm import joinedload
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.routing import BuildError
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

import os, glob, shutil, json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date
from io import BytesIO

from extensions import db, migrate
from models import Role, User, Prediction, PerformanceReview, ReviewStatus, Criteria, Appraisal, CycleEnrollment, RetentionPlan
from pm_reviews import bp as reviews_bp
from appraisal_routes import appraisal_bp
from forms import LoginForm


def _role_to_string(r) -> str:
    """Return a plain string for many role representations."""
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    if hasattr(r, "value") and isinstance(getattr(r, "value"), str):
        return r.value
    if hasattr(r, "name") and isinstance(getattr(r, "name"), str):
        return r.name
    for attr in ("name", "code", "title", "label", "role"):
        v = getattr(r, attr, None)
        if isinstance(v, str):
            return v
    return str(r)

def _current_role() -> str:
    """Normalized role as lowercase string: 'employee' | 'manager' | 'hr' | 'admin'."""
    s = ""
    try:
        if current_user and hasattr(current_user, "role"):
            s = _role_to_string(getattr(current_user, "role", None))
    except Exception:
        s = ""
    if not s and has_request_context():
        s = session.get("role", "")
    return (s or "").strip().lower()

# ----------------- Flask App (create BEFORE routes) -----------------
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '12345')
# --- Account blueprint (must exist before any @account_bp routes) ---
account_bp = Blueprint('account', __name__, url_prefix='/account')

# CSRF: disabled for simple HTML posts
csrf = CSRFProtect(app)
app.config["WTF_CSRF_ENABLED"] = False

# ---- SAFE URL helper to prevent BuildError in templates (added) ----
def safe_url_for(endpoint, **values):
    """Return a URL if endpoint exists; otherwise None. Use in templates to avoid 500s."""
    try:
        return url_for(endpoint, **values)
    except BuildError:
        return None

# register in Jinja
app.jinja_env.globals['safe_url_for'] = safe_url_for
def role_str(value):
    try:
        return (_role_to_string(value) or "").strip()
    except Exception:
        return str(value) if value is not None else ""

app.jinja_env.globals['role_str'] = role_str
app.jinja_env.filters['role_str'] = role_str
app.jinja_env.globals['role_str'] = _role_to_string

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
        row = {
            "timestamp": getattr(p, "timestamp", None),
            "result": getattr(p, "result", ""),
            "confidence": getattr(p, "confidence", ""),
        }
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
        row = {
            "timestamp": getattr(p, "timestamp", None),
            "result": getattr(p, "result", ""),
            "confidence": getattr(p, "confidence", ""),
        }
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

# Optional PDF export; will show a flash if reportlab isn’t installed
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# ---------- HR quick-action UI placeholders (safe app-level routes) -----

@app.route("/hr/cycle/new")
@login_required
def hr_ui_cycle_new():
    return render_template("hr_placeholder.html", title="Launch Cycle")

@app.route("/hr/reviewers")
@login_required
def hr_ui_assign_reviewers():
    return render_template("hr_placeholder.html", title="Assign Reviewers")

@app.route("/hr/progress")
@login_required
def hr_ui_progress():
    return render_template("hr_placeholder.html", title="Monitor Progress")

@app.route("/hr/reviews")
@login_required
def hr_ui_reviews():
    return render_template("hr_placeholder.html", title="HR Review / Approve / Reject")

@app.route("/hr/calibration")
@login_required
def hr_ui_calibrate():
    return render_template("hr_placeholder.html", title="Calibrate Ratings")

@app.route("/hr/communicate")
@login_required
def hr_ui_communicate():
    return render_template("hr_placeholder.html", title="Communicate Outcomes")

@app.route("/hr/export/pdf")
@login_required
def hr_export_pdf():
    if get_role(current_user) != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard_hr"))

    if not _PDF_OK:
        flash("PDF export needs 'reportlab'. Try CSV/Excel, or: pip install reportlab", "warning")
        return redirect(url_for("dashboard_hr"))

    total_employees = User.query.filter_by(role=Role.EMPLOYEE).count()
    total_appraisals = PerformanceReview.query.count()
    preds = Prediction.query.all()
    yes = sum(1 for p in preds if (p.result or "").lower() == "yes")
    yes_rate = (100.0 * yes / len(preds)) if preds else 0.0

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4); w, h = A4; y = h - 50
    c.setFont("Helvetica-Bold", 14); c.drawString(40, y, "HR Attrition Dashboard — Summary"); y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Total Employees: {total_employees}"); y -= 18
    c.drawString(40, y, f"Total Appraisals: {total_appraisals}"); y -= 18
    c.drawString(40, y, f"Total Predictions: {len(preds)}"); y -= 18
    c.drawString(40, y, f"Attrition Yes Rate: {yes_rate:.1f}%")
    c.showPage(); c.save(); buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"hr_summary_{datetime.utcnow():%Y%m%d}.pdf",
                     mimetype="application/pdf")

# Make {{ csrf_token() }} available in ALL templates
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

# === Database Config ===
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://attrition_user:12345@localhost:5432/attrition_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# === Mail Config ===
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# === Init Extensions & Blueprints ===
db.init_app(app)
migrate.init_app(app, db)
app.register_blueprint(reviews_bp)
app.register_blueprint(appraisal_bp)
# account_bp is registered after it’s declared below


# ✅ Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"   # redirect here if not logged in

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === Load Model & Encoder (ensure they are a matched pair) ===
def _list_with_ids(pattern, prefix):
    out = {}
    for p in glob.glob(pattern):
        name = os.path.basename(p)
        ident = name.replace(prefix, "").rsplit(".", 1)[0]
        out[ident] = p
    return out

def load_paired_model_and_encoder():
    """
    Prefer the root pair (model.pkl, encoder.pkl) saved together by train_model.py.
    If not found, pair timestamped files in /models by matching the same ID.
    """
    root_model = "model.pkl"
    root_encoder = "encoder.pkl"
    if os.path.exists(root_model) and os.path.exists(root_encoder):
        m, e = joblib.load(root_model), joblib.load(root_encoder)
        print("Loaded root pair: model.pkl + encoder.pkl")
        return m, e

    model_map = _list_with_ids("models/model_*.pkl", "model_")
    enc_map   = _list_with_ids("models/encoder_*.pkl", "encoder_")
    common_ids = set(model_map).intersection(enc_map)
    if not common_ids:
        print("WARNING: No matching model/encoder IDs found; falling back to independent latest files.")
        latest_model = max(glob.glob("models/model_*.pkl"), key=os.path.getctime)
        latest_encoder = max(glob.glob("models/encoder_*.pkl"), key=os.path.getctime)
        return joblib.load(latest_model), joblib.load(latest_encoder)

    newest_id = max(common_ids, key=lambda i: os.path.getctime(model_map[i]))
    print(f"Loaded paired files: {os.path.basename(model_map[newest_id])} + {os.path.basename(enc_map[newest_id])}")
    return joblib.load(model_map[newest_id]), joblib.load(enc_map[newest_id])

model, encoder = load_paired_model_and_encoder()

# === Helper: normalize role safely ===
def get_role(user):
    """Normalized role string: 'employee' | 'manager' | 'hr' | 'admin'."""
    try:
        val = getattr(user, "role", None)
        if isinstance(val, Role):
            return str(val.value).lower()
        return str(val).lower()
    except Exception:
        return ""


# === Columns (must match training) ===
CATEGORICAL_COLS = ['JobRole', 'Department', 'MaritalStatus', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField']
NUMERIC_COLS = [
    'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate',
    'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked',
    'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel',
    'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
    'YearsSinceLastPromotion','YearsWithCurrManager'
]
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
try:
    if os.path.exists("columns.joblib"):
        cols = joblib.load("columns.joblib")
        CATEGORICAL_COLS = list(cols.get("categorical_columns", CATEGORICAL_COLS))
        NUMERIC_COLS = list(cols.get("numerical_columns", NUMERIC_COLS))
        ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
        print("Loaded column order from columns.joblib")
except Exception as e:
    print("Warning: couldn't load columns.joblib:", e)

NUMERIC_COLS = [c for c in NUMERIC_COLS if isinstance(c, str)]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if isinstance(c, str)]

ENCODER_EXPECTED_INPUT = list(getattr(encoder, "feature_names_in_", CATEGORICAL_COLS))
print("Encoder expects categorical columns:", ENCODER_EXPECTED_INPUT, "(n =", len(ENCODER_EXPECTED_INPUT), ")")

def _build_encoder_input_df(form, encoder):
    if hasattr(encoder, "feature_names_in_"):
        enc_cols = list(encoder.feature_names_in_)
    else:
        enc_cols = CATEGORICAL_COLS

    row = {}
    cats = getattr(encoder, "categories_", None)

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
    if p >= 0.70:
        return "High", "danger"
    if p >= 0.40:
        return "Moderate", "warning"
    return "Low", "success"

def _build_top_drivers(x_row, feature_names, coef, k=8):
    rows = []
    for i, name in enumerate(feature_names):
        val = float(x_row[i])
        w = float(coef[i])
        contrib = val * w
        rows.append({
            "name": name,
            "value": val,
            "weight": w,
            "contribution": contrib,
            "direction": "↑ risk" if contrib > 0 else "↓ risk"
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows[:k]

def _recommendations(payload: dict, top_drivers: list, will_leave: bool):
    rec = []
    top_names = ", ".join(d["name"] for d in top_drivers[:3]) or "multiple factors"
    rec.append(f"Focus on: {top_names}.")

    if will_leave:
        try:
            js = int(payload.get("JobSatisfaction", 3))
            es = int(payload.get("EnvironmentSatisfaction", 3))
            wlb = int(payload.get("WorkLifeBalance", 3))
            dist = int(payload.get("DistanceFromHome", 0))
            yslp = int(payload.get("YearsSinceLastPromotion", 0))
            train = int(payload.get("TrainingTimesLastYear", 0))
            income = float(payload.get("MonthlyIncome", 0.0))
        except Exception:
            js = es = wlb = 3
            dist = yslp = train = 0
            income = 0.0

        if payload.get("OverTime") == "Yes":
            rec.append("Reduce sustained overtime; offer comp time or flexible scheduling.")
        if js <= 2:
            rec.append("Run a 1:1 to identify role fit issues; agree a 30-day motivation plan.")
        if es <= 2:
            rec.append("Address workspace/environmental irritants raised in prior feedback.")
        if wlb <= 2:
            rec.append("Improve work–life balance: enforce reasonable hours and encourage PTO.")
        if dist >= 20:
            rec.append("Consider hybrid/remote days or assignment closer to home.")
        if yslp >= 3:
            rec.append("Discuss growth; propose a stretch project and schedule a promotion review.")
        if train == 0:
            rec.append("Fund upskilling this quarter (formal course or certification).")
        if income and income < 4000:
            rec.append("Benchmark compensation vs pay band; adjust if materially below peers.")
    else:
        rec.append("Maintain regular recognition and quarterly growth check-ins.")
        rec.append("Keep workload steady; avoid prolonged overtime spikes.")

    return rec[:8]

# ----------------- ROUTES -----------------
@app.route('/')
def home():
    return render_template("home.html")

# ----------------- Authentication -----------------
from urllib.parse import urlparse, urljoin

def _is_safe_next(target: str) -> bool:
    if not target:
        return False
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return (test.scheme in ("http", "https")) and (ref.netloc == test.netloc)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()

    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        password   = (request.form.get("password")   or "").strip()
        expected   = (request.form.get("expected_role") or "").strip().upper() or None

        # find by username OR email
        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        open_target = f"login_{expected.lower()}" if expected else "login"

        # invalid credentials
        if not user or not check_password_hash(getattr(user, "password", "") or "", password):
            flash("Invalid username/email or password.", "danger")
            return redirect(url_for("home", open=open_target))

        # inactive
        if not getattr(user, "is_active", True):
            flash("Your account is pending Admin approval.", "warning")
            return redirect(url_for("home", open=open_target))

        # wrong portal (when expected_role is provided)
        if expected and get_role(user) != expected:
            friendly = expected.title()
            flash(f"This portal is for {friendly} accounts.", "danger")
            return redirect(url_for("home", open=open_target))

        # success
        login_user(user)
        session["user_id"]  = user.id
        session["username"] = user.username
        session["role"]     = get_role(user)

        nxt = request.args.get("next")
        if _is_safe_next(nxt):
            return redirect(nxt)
        return redirect(url_for("dashboard"))

    # GET
    return render_template("login.html", form=form)


# ---------- helpers ----------
def set_all_if_have(obj, value, *names):
    """Set value on every attribute that exists on obj from the provided names."""
    for n in names:
        if hasattr(obj, n):
            setattr(obj, n, (value or None))

# ----------------- Registration -----------------
# ----------------- Registration -----------------
@app.route('/register', methods=['POST'])
def register():
    # Core fields
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    role = (request.form.get('role', '').strip() or '').upper()

    # Employee profile fields — accept multiple input names from the form
    f = request.form
    name = (f.get('name') or f.get('full_name') or f.get('fullname') or '').strip()
    employee_id = (f.get('employee_id') or f.get('emp_id') or f.get('employeeID') or '').strip()
    department = (f.get('department') or f.get('dept') or f.get('department_name') or '').strip()
    title = (f.get('title') or f.get('job_title') or f.get('jobTitle') or '').strip()
    manager_name = (f.get('manager_name') or f.get('manager') or f.get('supervisor_name') or '').strip()
    phone = (f.get('phone') or f.get('phone_number') or f.get('mobile') or '').strip()
    location = (f.get('location') or f.get('office') or f.get('office_location') or f.get('city') or '').strip()

    # validations ...
    if not username or not email or not password or not confirm_password or not role:
        flash('Please fill in all required fields.', 'danger')
        return redirect(url_for('home', open=f'register_{(role or "EMPLOYEE").lower()}'))
    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('home', open=f'register_{role.lower()}'))
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('home', open=f'register_{role.lower()}'))
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'danger')
        return redirect(url_for('home', open=f'register_{role.lower()}'))

    if role == 'EMPLOYEE':
        missing = [lbl for lbl, val in [
            ('Full Name', name),
            ('Employee ID', employee_id),
            ('Department', department),
            ('Job Title', title),
            ('Manager', manager_name),
            ('Phone', phone),
            ('Location', location),
        ] if not val]
        if missing:
            flash('Please complete: ' + ', '.join(missing) + '.', 'danger')
            return redirect(url_for('home', open='register_employee'))

    hashed_pw = generate_password_hash(password)

    # Create user and WRITE TO REAL COLUMNS (no set_all_if_have)
    new_user = User(
        username=username,
        email=email,
        password=hashed_pw,
        role=Role(role),
        is_active=False
    )
    new_user.full_name    = name
    new_user.employee_id  = employee_id
    new_user.department   = department
    new_user.job_title    = title
    new_user.manager_name = manager_name
    new_user.phone        = phone
    new_user.location     = location

    db.session.add(new_user)
    db.session.commit()

    # Notify admins (best-effort)
    try:
        admins = User.query.filter_by(role=Role.ADMIN, is_active=True).all()
        if admins:
            admin_emails = [a.email for a in admins]
            msg = Message("New User Pending Approval", recipients=admin_emails)
            details = [
                f"Username: {username}",
                f"Email: {email}",
                f"Role: {role}",
            ]
            if role == 'EMPLOYEE':
                details += [
                    f"Name: {name}",
                    f"Employee ID: {employee_id}",
                    f"Department: {department}",
                    f"Title: {title}",
                    f"Manager: {manager_name}",
                    f"Phone: {phone}",
                    f"Location: {location}",
                ]
            msg.body = "A new user has registered:\n\n" + "\n".join(details)
            mail.send(msg)
    except Exception as e:
        print("Email send failed:", e)

    flash('Registration successful. Awaiting Admin approval.', 'info')
    return redirect(url_for('home', open=f'login_{role.lower()}'))

# --- Account/Profile endpoints ---
# --- Account/Profile endpoints ---
@account_bp.post('/profile/update', endpoint='update_profile')
@login_required
def update_profile():
    f = request.form

    # Pull incoming fields (supporting multiple input names)
    nm   = (f.get('name') or f.get('full_name') or f.get('fullname') or current_user.full_name)
    dept = (f.get('department') or f.get('dept') or f.get('department_name') or current_user.department)
    ttl  = (f.get('title') or f.get('job_title') or f.get('jobTitle') or current_user.job_title)
    mng  = (f.get('manager_name') or f.get('manager') or f.get('supervisor_name') or current_user.manager_name)
    ph   = (f.get('phone') or f.get('phone_number') or f.get('mobile') or current_user.phone)
    loc  = (f.get('location') or f.get('office') or f.get('office_location') or f.get('city') or current_user.location)
    eid  = (f.get('employee_id') or f.get('emp_id') or f.get('employeeID') or current_user.employee_id or '').strip()

    # WRITE TO REAL COLUMNS
    current_user.full_name    = nm
    current_user.department   = dept
    current_user.job_title    = ttl
    current_user.manager_name = mng
    current_user.phone        = ph
    current_user.location     = loc
    if eid:
        current_user.employee_id = eid

    # Optional password change
    pw  = f.get('password') or ''
    pw2 = f.get('password_confirm') or ''
    if pw or pw2:
        if pw != pw2:
            msg = 'Passwords do not match.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(ok=False, message=msg), 400
            flash(msg, 'danger')
            return redirect(request.referrer or url_for('appraisal.employee_list_appraisals'))
        if len(pw) < 8:
            msg = 'Password must be at least 8 characters.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(ok=False, message=msg), 400
            flash(msg, 'danger')
            return redirect(request.referrer or url_for('appraisal.employee_list_appraisals'))
        current_user.password = generate_password_hash(pw)

    db.session.commit()

    # AJAX response for the modal
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(
            ok=True,
            message='Profile updated.',
            user={
                "name": current_user.full_name,
                "department": current_user.department,
                "title": current_user.job_title,
                "manager_name": current_user.manager_name,
                "phone": current_user.phone,
                "location": current_user.location,
                "employee_id": current_user.employee_id,
                "role": str(getattr(getattr(current_user, "role", ""), "value", getattr(current_user, "role", ""))),
            }
        )

    flash('Profile updated.', 'success')
    return redirect(request.referrer or url_for('appraisal.employee_list_appraisals'))

app.register_blueprint(account_bp)


@app.route('/recover', methods=['GET', 'POST'])
def recover():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            token = serializer.dumps(email, salt='password-recover')
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message("Password Reset Request", recipients=[email])
            msg.body = f"Click the link to reset your password:\n{reset_url}\n\nThis link is valid for 1 hour."
            mail.send(msg)
            flash("Password reset link sent to your email.", "info")
            return redirect(url_for('home', open='login'))
        else:
            flash("No account found with that email.", "danger")

    return render_template('recover.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
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
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(request.url)

        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("Your password has been updated. Please log in.", "success")
            return redirect(url_for('home', open='login'))

    return render_template('reset_password.html', token=token)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home', open='login'))

# ----------------- Dashboards -----------------
@app.route("/dashboard")
@login_required
def dashboard():
    role = _current_role()
    if role == "employee":
        return redirect(url_for("dashboard_employee"))
    if role == "manager":
        return redirect(url_for("dashboard_manager"))
    if role == "hr":
        return redirect(url_for("dashboard_hr"))
    if role == "admin":
        return redirect(url_for("dashboard_admin"))
    return render_template("dashboard_generic.html")

@app.route("/dashboard/employee")
@login_required
def dashboard_employee():
    role = _current_role()
    if role == "manager":
        return redirect(url_for("dashboard_manager"))
    if role == "hr":
        return redirect(url_for("dashboard_hr"))
    if role == "admin":
        return redirect(url_for("dashboard_admin"))

    # 1) Make sure the session isn't stuck in a failed transaction
    try:
        db.session.rollback()
    except Exception:
        pass

    ctx = dict(
        active_appraisal=None,
        self_progress=0,
        section_progress=[],
        goals=[],
        achievements=[],
        feedback=[],
        announcements=[],
        appraisals_count=0,
        current_cycle=None,
        appraisal_deadline=None,
    )

    try:
        current_cycle = get_current_cycle() if "get_current_cycle" in globals() else None
        ctx["current_cycle"] = current_cycle

        # 2) Eager-load relationships so the template won't lazy-load
        q = (Appraisal.query
                .options(
                    joinedload(Appraisal.workload),
                    joinedload(Appraisal.scores)
                )
                .filter_by(employee_id=current_user.id))

        # If your model uses cycle_id instead of 'cycle', adjust this line accordingly
        if current_cycle:
            q = q.filter_by(cycle=current_cycle)

        active = q.order_by(Appraisal.created_at.desc()).first()
        ctx["active_appraisal"] = active

        if active:
            ctx["self_progress"] = getattr(active, "progress", 0)
            ctx["section_progress"] = getattr(active, "section_progress", [])
            if getattr(active, "deadline", None):
                ctx["appraisal_deadline"] = active.deadline

        # Be tolerant of Enum or string statuses
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
        # Ensure the session is usable after any error
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

    # helper: DISTINCT count by status using the same join logic
    def _count_by_status(*statuses):
        q = (db.session.query(func.count(func.distinct(Appraisal.id)))
             .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
             .filter(or_(Appraisal.supervisor_id == mid,
                         CycleEnrollment.manager_id == mid)))
        if len(statuses) == 1:
            q = q.filter(Appraisal.status == statuses[0])
        else:
            q = q.filter(Appraisal.status.in_(statuses))
        return q.scalar() or 0

    pending_count     = _count_by_status(ReviewStatus.SUBMITTED_TO_MANAGER)
    awaiting_hr_count = _count_by_status(ReviewStatus.MANAGER_SUBMITTED)
    returned_count    = _count_by_status(ReviewStatus.REJECTED)  # add other "returned" statuses if you have them
    completed_count   = _count_by_status(ReviewStatus.HR_REVIEWED, ReviewStatus.APPROVED)

    # Distinct active employees mapped to me (already DISTINCT here)
    team_members = (db.session.query(func.count(func.distinct(User.id)))
                    .join(CycleEnrollment, CycleEnrollment.employee_id == User.id)
                    .filter(CycleEnrollment.manager_id == mid, User.is_active.is_(True))
                    .scalar()) or 0


    # Helpful log so you can verify numbers in the console
    app.logger.info("MgrCards mid=%s pending=%s awaiting=%s returned=%s completed=%s team=%s",
                    mid, pending_count, awaiting_hr_count, returned_count, completed_count, team_members)

    # Provide MANY aliases so any template naming works
    stats = dict(
        team_members=team_members,
        pending=pending_count,
        awaiting_hr=awaiting_hr_count,
        returned=returned_count,
        completed=completed_count,
    )
    ctx = {
        # primary vars
        "pending_count": pending_count,
        "awaiting_hr_count": awaiting_hr_count,
        "returned_count": returned_count,
        "completed_count": completed_count,
        "team_members": team_members,
        "stats": stats,
        # common aliases used in templates
        "pending": pending_count,
        "pending_total": pending_count,
        "pending_appraisals": pending_count,
        "awaiting_hr": awaiting_hr_count,
        "awaiting": awaiting_hr_count,
        "submitted_count": awaiting_hr_count,
        "completed": completed_count,
        "completed_total": completed_count,
        "team_count": team_members,
        "members_count": team_members,
        "cards": stats,
        "metrics": stats,
    }

    return render_template("manager_dashboard.html", **ctx)

@app.route('/dashboard/hr')
@login_required
def dashboard_hr():
    """
    HR dashboard:
      • Attrition analytics (unchanged)
      • FIXED: Appraisal counters + queue now use Appraisal/ReviewStatus and
               provide hr_awaiting_count, hr_returned_count, hr_finalized_count,
               hr_overdue_count, hr_queue, recent_appraisals.
    """
    if get_role(current_user) != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_hr"))

    def sfloat(x):
        try: return float(x)
        except: return None

    def as_dict(payload):
        if isinstance(payload, dict):
            return payload
        try:
            return json.loads(payload or "{}")
        except Exception:
            return {}

    # ----- People & predictions (existing logic kept) -----
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
            # keep compatibility with old table if any
            status_dist[st.name] = PerformanceReview.query.filter_by(status=st).count()
    except Exception:
        status_dist = {}

    try:
        preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5000).all()
    except Exception:
        preds = []

    yes_count = 0
    risk_over_time = {}
    dept_stats = {}
    ot_stats = {"Yes": {"yes": 0, "tot": 0}, "No": {"yes": 0, "tot": 0}}
    tenure_bins = {"<1y": {"yes": 0, "tot": 0},"1–3y": {"yes": 0, "tot": 0},"3–5y": {"yes": 0, "tot": 0},"5–10y": {"yes": 0, "tot": 0},"10y+": {"yes": 0, "tot": 0}}
    recent_rows = []

    for p in preds:
        ts = getattr(p, "timestamp", None)
        proba = sfloat(getattr(p, "confidence", None))
        result_text = (getattr(p, "result", "") or "").strip()
        data = as_dict(getattr(p, "input_data", {}))

        recent_rows.append({
            "ts": ts.strftime("%Y-%m-%d %H:%M") if ts else "",
            "result": result_text,
            "proba": f"{(proba or 0)*100:.1f}%",
            "dept": data.get("Department", ""),
            "role": data.get("JobRole", ""),
            "ot": data.get("OverTime", ""),
            "years": data.get("YearsAtCompany", ""),
        })

        if result_text.lower() == "yes":
            yes_count += 1
        if ts and proba is not None:
            risk_over_time.setdefault(ts.date().isoformat(), []).append(proba)

        dept = data.get("Department", "Unknown")
        dept_stats.setdefault(dept, {"yes": 0, "tot": 0})
        dept_stats[dept]["tot"] += 1
        if result_text.lower() == "yes":
            dept_stats[dept]["yes"] += 1

        ot = str(data.get("OverTime", "Unknown"))
        if ot in ot_stats:
            ot_stats[ot]["tot"] += 1
            if result_text.lower() == "yes":
                ot_stats[ot]["yes"] += 1

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
        if result_text.lower() == "yes":
            tenure_bins[b]["yes"] += 1

    total_predictions = len(preds)
    yes_rate = round(100.0 * yes_count / total_predictions, 1) if total_predictions else 0.0

    trend_dates = sorted(risk_over_time.keys())
    trend_avg = [round(100.0 * (sum(risk_over_time[d]) / max(len(risk_over_time[d]), 1)), 2) for d in trend_dates]

    dept_labels, dept_rates = [], []
    for d, obj in sorted(dept_stats.items(), key=lambda kv: kv[0]):
        rate = 100.0 * obj["yes"] / obj["tot"] if obj["tot"] else 0.0
        dept_labels.append(d)
        dept_rates.append(round(rate, 2))

    ot_labels = list(ot_stats.keys())
    ot_rates = [round(100.0 * v["yes"] / v["tot"], 2) if v["tot"] else 0.0 for v in ot_stats.values()]

    tenure_labels = list(tenure_bins.keys())
    tenure_rates = [round(100.0 * v["yes"] / v["tot"], 2) if v["tot"] else 0.0 for v in tenure_bins.values()]

    # ----- FIXED: Appraisal section built from Appraisal model -----
    # Awaiting HR queue (manager-submitted)
    awaiting_rows = (Appraisal.query
                     .filter(Appraisal.status == ReviewStatus.MANAGER_SUBMITTED)
                     .order_by(Appraisal.created_at.desc())
                     .all())

    # Counts for KPI cards
    hr_awaiting_count = len(awaiting_rows)
    hr_returned_count = (db.session.query(func.count(Appraisal.id))
                         .filter(Appraisal.status == ReviewStatus.REJECTED).scalar() or 0)
    hr_finalized_count = (db.session.query(func.count(Appraisal.id))
                          .filter(Appraisal.status == ReviewStatus.APPROVED).scalar() or 0)
    hr_overdue_count = (db.session.query(func.count(Appraisal.id))
                        .filter(Appraisal.period_end != None,
                                Appraisal.period_end < date.today(),
                                Appraisal.status != ReviewStatus.APPROVED).scalar() or 0)

    # Status distribution from Appraisal for donut chart
    status_pairs = (db.session.query(Appraisal.status, func.count(Appraisal.id))
                    .group_by(Appraisal.status).all())
    status_labels = [getattr(s, "value", str(s)) for s, _ in status_pairs]
    status_counts = [c for _, c in status_pairs]

    # Lookup names
    emp_ids = {a.employee_id for a in awaiting_rows}
    mgr_ids = {a.supervisor_id for a in awaiting_rows}
    # also for recent finalized list
    order_col = getattr(Appraisal, "updated_at", getattr(Appraisal, "created_at", Appraisal.id))
    recent_approved = (Appraisal.query
                       .filter(Appraisal.status == ReviewStatus.APPROVED)
                       .order_by(order_col.desc()).limit(12).all())
    emp_ids |= {a.employee_id for a in recent_approved}
    mgr_ids |= {a.supervisor_id for a in recent_approved}

    ids = list(emp_ids | mgr_ids)
    user_map = {u.id: u for u in (User.query.filter(User.id.in_(ids)).all() if ids else [])}

    # Queue payload for template
    hr_queue = []
    for a in awaiting_rows:
        hr_queue.append({
            "id": a.id,
            "employee_id": a.employee_id,
            "employee_name": getattr(user_map.get(a.employee_id), "username", None),
            "manager_id": a.supervisor_id,
            "manager_name": getattr(user_map.get(a.supervisor_id), "username", None),
            "period_start": a.period_start,
            "period_end": a.period_end,
            "total_score": a.total_score,
            "status": getattr(a.status, "value", str(a.status)),
            # Buttons are links to the queue UI (your POST action buttons live there)
            "review_url": url_for("appraisal.hr_review_queue_ui"),
            "return_url": url_for("appraisal.hr_review_queue_ui"),
            "finalize_url": url_for("appraisal.hr_review_queue_ui"),
        })

    # Recent finalized (right-hand table)
    recent_appraisals = [{
        "id": a.id,
        "employee_id": a.employee_id,
        "employee_name": getattr(user_map.get(a.employee_id), "username", None),
        "period_start": a.period_start,
        "period_end": a.period_end,
        "total_score": a.total_score,
        "finalized_at": getattr(a, "updated_at", None),
    } for a in recent_approved]

    # optional: selected cycle id from querystring for the filter
    selected_cycle_id = request.args.get("cycle_id")

    # Prefer the real Appraisal count for display
    try:
        total_appraisals_real = db.session.query(func.count(Appraisal.id)).scalar() or 0
    except Exception:
        total_appraisals_real = total_appraisals or 0

    return render_template(
        "hr_dashboard.html",
        # Predictions area
        total_employees=total_employees or 0,
        total_appraisals=total_appraisals_real,   # <-- use Appraisal count
        appraisals_total=total_appraisals_real,
        total_predictions=total_predictions or 0,
        yes_rate=yes_rate or 0.0,
        avg_conf_yes=0.0,
        trend_dates=trend_dates or [],
        trend_avg=trend_avg or [],
        dept_labels=dept_labels or [],
        dept_rates=dept_rates or [],
        ot_labels=ot_labels or [],
        ot_rates=ot_rates or [],
        tenure_labels=tenure_labels or [],
        tenure_rates=tenure_rates or [],
        recent_rows=recent_rows[:12] or [],
        # Appraisals area (fixed)
        hr_awaiting_count=hr_awaiting_count,
        hr_returned_count=hr_returned_count,
        hr_finalized_count=hr_finalized_count,
        hr_overdue_count=hr_overdue_count,
        hr_queue=hr_queue,
        status_labels=status_labels or [],
        status_counts=status_counts or [],
        recent_appraisals=recent_appraisals or [],
        selected_cycle_id=selected_cycle_id
    )

@app.context_processor
def inject_now():
    return {"current_year": datetime.utcnow().year}

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

        categorical_columns = ['JobRole', 'Department', 'MaritalStatus',
                               'Gender', 'OverTime', 'BusinessTravel', 'EducationField']
        numerical_columns = [col for col in X.columns if col not in categorical_columns]

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(X_train[categorical_columns])

        X_train_enc = encoder.transform(X_train[categorical_columns])
        X_test_enc  = encoder.transform(X_test[categorical_columns])

        X_train_final = np.hstack((X_train[numerical_columns].values, X_train_enc))
        X_test_final  = np.hstack((X_test[numerical_columns].values, X_test_enc))

        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            C=0.5,
            random_state=42
        )
        model.fit(X_train_final, y_train)

        os.makedirs("models", exist_ok=True)
        ts = int(datetime.utcnow().timestamp())
        model_file   = os.path.join("models", f"model_{ts}.pkl")
        encoder_file = os.path.join("models", f"encoder_{ts}.pkl")
        cols_file    = os.path.join("models", "columns.joblib")

        joblib.dump(model, model_file)
        joblib.dump(encoder, encoder_file)
        joblib.dump(
            {"categorical_columns": categorical_columns, "numerical_columns": numerical_columns},
            cols_file
        )

        flash("✅ Model retrained and saved successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"⚠️ Training failed: {e}", "danger")

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

    models = [
        {"name": os.path.basename(f), "ctime": os.path.getctime(f)}
        for f in model_files
    ]
    encoders = [
        {"name": os.path.basename(f), "ctime": os.path.getctime(f)}
        for f in encoder_files
    ]

    return render_template(
        "admin_dashboard.html",
        users=users,
        models=models,
        encoders=encoders,
        criteria=criteria
    )

@app.route('/save_retention_plan', methods=['POST'])
@login_required
def save_retention_plan():
    try:
        user_id = current_user.id
        immediate = request.form.get("immediate_plan", "").strip()
        next_steps = request.form.get("next_steps", "").strip()
        assigned = request.form.get("assigned_to", "").strip()

        if not immediate:
            flash("Immediate plan is required.", "danger")
            return redirect(url_for("predict"))

        plan = RetentionPlan(
            user_id=user_id,
            immediate_plan=immediate,
            next_steps=next_steps,
            assigned_to=assigned
        )
        db.session.add(plan)
        db.session.commit()

        flash("✅ Retention plan saved successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"⚠️ Error saving plan: {str(e)}", "danger")

    return redirect(url_for("prediction_history"))

from flask import g

@app.context_processor
def inject_notifications():
    """Inject unread notifications count into all templates."""
    count = 0
    items = []
    if current_user.is_authenticated:
        # Example: notifications from predictions or appraisals
        try:
            if get_role(current_user) in ["MANAGER", "HR"]:
                # Unseen predictions
                count = Prediction.query.filter_by(user_id=current_user.id).count()
                items = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()
            if get_role(current_user) == "EMPLOYEE":
                # Pending appraisals
                count = Appraisal.query.filter_by(employee_id=current_user.id, status=ReviewStatus.DRAFT).count()
        except Exception as e:
            app.logger.warning("inject_notifications failed: %s", e)
            db.session.rollback()
    return dict(notif_count=count, notif_items=items)



# ----------------- Prediction -----------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
@csrf.exempt
def predict():
    if request.method == 'GET':
        return render_template('predict_form.html')

    app.logger.info('POST /predict data: %s', dict(request.form))
    form = request.form

    missing_numeric = [c for c in NUMERIC_COLS if form.get(c, "") == ""]
    if missing_numeric:
        flash(f"Missing numeric fields: {', '.join(missing_numeric)}", 'danger')
        return redirect(url_for('predict'))

    try:
        x_num = np.array([float(form[col]) for col in NUMERIC_COLS], dtype=float).reshape(1, -1)

        enc_df, enc_cols = _build_encoder_input_df(form, encoder)
        x_enc = encoder.transform(enc_df)

        n_model_in = getattr(model, 'n_features_in_', None)

        if n_model_in is not None and x_enc.shape[1] == n_model_in:
            x_final = x_enc
        elif n_model_in is not None and (x_enc.shape[1] + x_num.shape[1]) == n_model_in:
            x_final = np.hstack([x_num, x_enc])
        else:
            if set(enc_cols) == set(CATEGORICAL_COLS):
                x_final = np.hstack([x_num, x_enc])
            else:
                x_final = np.hstack([x_num, x_enc])

        proba = model.predict_proba(x_final)[0][1]
        pred = (proba >= 0.5)

        payload = {k: form.get(k, "") for k in (NUMERIC_COLS + enc_cols)}
        rec = Prediction(
            user_id=current_user.id,
            input_data=payload,
            result='Yes' if pred else 'No',
            confidence=f"{proba:.3f}"
        )
        db.session.add(rec)
        db.session.commit()

        try:
            feature_names_enc = list(encoder.get_feature_names_out(enc_cols))
        except Exception:
            feature_names_enc = list(getattr(encoder, "get_feature_names", lambda x: x)(enc_cols))

        used_enc_only = (x_final.shape[1] == x_enc.shape[1])
        if used_enc_only:
            feature_names = feature_names_enc
        else:
            feature_names = NUMERIC_COLS + feature_names_enc

        coef = model.coef_[0][:len(feature_names)]
        x_row = x_final[0]
        top_drivers = _build_top_drivers(x_row, feature_names, coef, k=8)

        risk_label, risk_color = _risk_bucket(proba)
        will_leave = bool(pred)
        recs = _recommendations(payload, top_drivers, will_leave)

        result_ctx = {
            "prediction_text": 'Employee will Attrit' if pred else 'Employee will NOT Attrit',
            "probability": f"{proba*100:.1f}%",
            "pred": 'Yes' if pred else 'No',
            "proba": proba,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "top_drivers": top_drivers,
            "recommendations": recs,
            "input_data": payload,
            "InputData": payload,
            "Result": 'Yes' if pred else 'No',
            "Confidence": f"{proba:.3f}",
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return render_template('result.html', **result_ctx, results=[result_ctx])

    except Exception as e:
        db.session.rollback()
        flash(f"Prediction error: {e}", 'danger')
        return redirect(url_for('predict'))

@app.route("/predict_form")
@login_required
def predict_form():
    return redirect(url_for("predict"))

@app.route('/admin/clear_predictions', methods=['POST'])
@login_required
def clear_predictions():
    if get_role(current_user) != "admin":
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

        # Allow only the owner (or admin/HR) to delete
        if pred.user_id != current_user.id and get_role(current_user) not in ["ADMIN", "HR"]:
            flash("You don’t have permission to delete this prediction.", "danger")
            return redirect(url_for('prediction_history'))

        db.session.delete(pred)
        db.session.commit()
        flash("Prediction deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting prediction: {e}", "danger")

    return redirect(url_for('prediction_history'))


# ----------------- Admin Actions -----------------
@app.route('/admin/create_user', methods=['POST'])
@login_required
def create_user():
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')

    if not username or not email or not password or not role:
        flash("All fields are required", "danger")
        return redirect(url_for('dashboard_admin'))

    if User.query.filter_by(username=username).first():
        flash("Username already exists", "danger")
        return redirect(url_for('dashboard_admin'))

    if User.query.filter_by(email=email).first():
        flash("Email already registered", "danger")
        return redirect(url_for('dashboard_admin'))

    new_user = User(
        username=username,
        email=email,
        password=generate_password_hash(password),
        role=Role(role),
        is_active=True
    )
    db.session.add(new_user)
    db.session.commit()

    flash(f"{role.title()} account '{username}' created successfully!", "success")
    return redirect(url_for('dashboard_admin'))

@app.route('/admin/approve/<int:user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    user = User.query.get_or_404(user_id)
    user.is_active = True
    db.session.commit()
    flash(f"✅ User {user.username} has been approved.", "success")
    return redirect(url_for('dashboard_admin'))

@app.route('/admin/reject/<int:user_id>', methods=['POST'])
@login_required
def reject_user(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f"❌ User {user.username} has been rejected and removed.", "danger")
    return redirect(url_for('dashboard_admin'))

@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@login_required
def reset_user_password(user_id):
    if get_role(current_user) != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_admin"))

    user = User.query.get_or_404(user_id)
    new_pw = "changeme123"
    user.password = generate_password_hash(new_pw)
    db.session.commit()
    flash(f"🔑 Password for {user.username} has been reset to '{new_pw}'.", "warning")
    return redirect(url_for('dashboard_admin'))

from sqlalchemy.exc import IntegrityError
from flask import request, flash, redirect, url_for

# ... your imports above ...

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
# @admin_required  # keep your own guard if you have one
def delete_user(user_id):
    u = User.query.get_or_404(user_id)

    # Count references from appraisals (as employee & as supervisor)
    emp_refs = Appraisal.query.filter_by(employee_id=user_id).count()
    sup_refs = Appraisal.query.filter_by(supervisor_id=user_id).count()

    if emp_refs or sup_refs:
        # Safer default: deactivate instead of hard delete
        # (assumes your User has an 'is_active' boolean column)
        u.is_active = False
        try:
            db.session.commit()
            flash(
                f"User '{getattr(u, 'username', user_id)}' is referenced by "
                f"{emp_refs} appraisal(s) as employee and {sup_refs} as supervisor. "
                f"The account was deactivated instead of deleted.",
                "warning",
            )
        except IntegrityError:
            db.session.rollback()
            flash("Could not update user status.", "danger")
        return redirect(request.referrer or url_for("dashboard"))

    # Not referenced anywhere → safe to hard delete
    try:
        db.session.delete(u)
        db.session.commit()
        flash("User deleted.", "success")
    except IntegrityError:
        db.session.rollback()
        # Fallback: deactivate if a late FK shows up
        u.is_active = False
        db.session.commit()
        flash(
            "User had linked data and could not be hard-deleted. The account was deactivated instead.",
            "warning",
        )

    return redirect(request.referrer or url_for("dashboard"))


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
                if f != latest_model:
                    os.remove(f)

        if encoder_files:
            latest_encoder = max(encoder_files, key=os.path.getctime)
            for f in encoder_files:
                if f != latest_encoder:
                    os.remove(f)

        flash("🧹 Old models and encoders deleted successfully, only latest kept.", "success")
    except Exception as e:
        flash(f"⚠️ Error deleting old models: {e}", "danger")

    return redirect(url_for('dashboard_admin'))

# ----------------- Run -----------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
