# appraisal_routes.py
import io
from datetime import date, timedelta, datetime

import pandas as pd
from flask import Blueprint, render_template, redirect, url_for, flash, request, send_file, session
from flask_login import login_required, current_user
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from extensions import db
from models import (
    Appraisal, Workload, Score, Criteria,
    User, AppraisalCycle, CycleEnrollment,
    ReviewStatus,
)
from appraisal_forms import AppraisalForm

appraisal_bp = Blueprint("appraisal", __name__, url_prefix="/appraisal")

# -------------------------------
# Role helpers
# -------------------------------
from functools import wraps

def _role_upper():
    """Normalize current user's role (Enum|string|session) to UPPER string."""
    rv = getattr(current_user, "role", None)
    if hasattr(rv, "value"):  # Enum case
        rv = rv.value
    if rv is None:
        rv = session.get("role")
    return (str(rv) if rv is not None else "").upper()

def hr_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if _role_upper() != "HR":
            flash("HR access only.", "danger")
            return redirect(url_for("dashboard"))
        return fn(*args, **kwargs)
    return wrapper

def role_required(*roles):
    target = {r.upper() for r in roles}
    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapper(*args, **kwargs):
            if _role_upper() not in target:
                flash("Access denied.", "danger")
                return redirect(url_for("dashboard"))
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def _users_by_role(role_str):
    # Works even if column is Enum(Role)
    return (User.query.filter_by(role=role_str, is_active=True)
            .order_by(User.username.asc()).all())

def _attach_people(rows):
    """Attach .employee and .supervisor objects for display."""
    ids = set()
    for a in rows:
        ids.add(a.employee_id)
        ids.add(a.supervisor_id)
    umap = {u.id: u for u in (User.query.filter(User.id.in_(ids)).all() if ids else [])}
    for a in rows:
        a.employee = umap.get(a.employee_id)
        a.supervisor = umap.get(a.supervisor_id)
    return rows

# -------------------------------
# HR: Create Cycle
# -------------------------------
@appraisal_bp.route('/cycle/new', methods=['GET', 'POST'])
@login_required
@hr_required
def new_cycle():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        start_date = request.form.get('start_date') or date.today().isoformat()
        end_date   = request.form.get('end_date')   or (date.today()+timedelta(days=90)).isoformat()
        desc = (request.form.get('description') or '').strip()

        # sections
        sec_names   = request.form.getlist('section_name[]')
        sec_weights = request.form.getlist('section_weight[]')
        sections, total_weight = [], 0.0
        for n, w in zip(sec_names, sec_weights):
            n = (n or '').strip()
            if not n:
                continue
            try:
                w_val = float(w)
            except Exception:
                w_val = 0.0
            total_weight += w_val
            sections.append({"name": n, "weight": round(w_val/100.0, 4)})

        # scale
        scale_type = request.form.get('scale_type', 'numeric')
        if scale_type == 'numeric':
            try:
                s_min = int(request.form.get('scale_min', 1))
                s_max = int(request.form.get('scale_max', 5))
                s_step = int(request.form.get('scale_step', 1))
            except Exception:
                s_min, s_max, s_step = 1, 5, 1
            scale = list(range(s_min, s_max+1, s_step))
        else:
            labels = (request.form.get('scale_labels') or '')
            scale = [x.strip() for x in labels.split(',') if x.strip()]

        # participants (employee -> manager)
        mappings = []
        for emp_id in request.form.getlist('participants[]'):
            mgr_id = request.form.get(f'manager_for__{emp_id}')
            if mgr_id:
                mappings.append((int(emp_id), int(mgr_id)))

        action = request.form.get('action', 'draft')  # 'draft' | 'open'

        # validation
        errors = []
        if not name: errors.append('Name is required.')
        try:
            sd = datetime.fromisoformat(start_date).date()
            ed = datetime.fromisoformat(end_date).date()
            if ed < sd: errors.append('End date must be after start date.')
        except Exception:
            errors.append('Invalid dates.')
        if not sections: errors.append('Add at least one section.')
        if abs(total_weight - 100.0) > 0.01: errors.append('Section weights must total 100%.')
        if not scale: errors.append('Provide a rating scale.')
        if not mappings: errors.append('Add at least one participant and assign a manager.')

        dup = AppraisalCycle.query.filter(db.func.lower(AppraisalCycle.name) == name.lower()).first()
        if dup:
            errors.append('A cycle with that name already exists. Choose a different name or open the existing cycle.')

        if errors:
            for e in errors: flash(e, 'danger')
            return render_template('appraisal/new_cycle.html',
                                   employees=_users_by_role('EMPLOYEE'),
                                   managers=_users_by_role('MANAGER'))

        # create cycle
        template = {"sections": sections, "scale": scale, "description": desc}
        cycle = AppraisalCycle(
            name=name, start_date=sd, end_date=ed,
            status='Open' if action == 'open' else 'Draft',
            template=template
        )
        db.session.add(cycle)

        try:
            db.session.flush()
        except IntegrityError:
            db.session.rollback()
            flash('That cycle name already exists. Please pick a unique name.', 'danger')
            return render_template('appraisal/new_cycle.html',
                                   employees=_users_by_role('EMPLOYEE'),
                                   managers=_users_by_role('MANAGER'))

        # enroll (CycleEnrollment)
        created = 0
        for emp_id, mgr_id in mappings:
            exists = CycleEnrollment.query.filter_by(employee_id=emp_id, cycle_id=cycle.id).first()
            if exists:
                continue
            db.session.add(CycleEnrollment(
                employee_id=emp_id, manager_id=mgr_id,
                cycle_id=cycle.id, status='Draft'
            ))
            created += 1

        db.session.commit()
        flash(f'Cycle “{name}” created ({cycle.status}). Participants enrolled: {created}.', 'success')
        return redirect(url_for('appraisal.cycle_detail', cycle_id=cycle.id))

    # GET
    return render_template('appraisal/new_cycle.html',
                           employees=_users_by_role('EMPLOYEE'),
                           managers=_users_by_role('MANAGER'))


@appraisal_bp.route('/cycles/<int:cycle_id>')
@login_required
@hr_required
def cycle_detail(cycle_id):
    cycle = AppraisalCycle.query.get_or_404(cycle_id)
    enrollments = CycleEnrollment.query.filter_by(cycle_id=cycle.id).all()
    stats = {
        "participants": len(enrollments),
        "managers": len({e.manager_id for e in enrollments}),
        "status": cycle.status,
        "start": cycle.start_date,
        "end": cycle.end_date,
    }
    return render_template('appraisal/cycle_detail.html',
                           cycle=cycle, stats=stats, appraisals=enrollments)

# -----------------------------
# Criteria Admin
# -----------------------------
@appraisal_bp.route("/admin/criteria", methods=["GET", "POST"])
@login_required
def admin_manage_criteria():
    if _role_upper() != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name")
        type_ = request.form.get("type")
        if name and type_:
            db.session.add(Criteria(name=name, type=type_))
            db.session.commit()
            flash("Criteria added successfully!", "success")
        else:
            flash("Please provide valid input", "danger")

    criteria = Criteria.query.all()
    return render_template("appraisal/admin_criteria.html", criteria=criteria, role="admin")


@appraisal_bp.route("/admin/criteria/delete/<int:criteria_id>", methods=["POST"])
@login_required
def delete_criteria(criteria_id):
    if _role_upper() != "ADMIN":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    criteria = Criteria.query.get_or_404(criteria_id)
    db.session.delete(criteria)
    db.session.commit()
    flash("Criteria deleted!", "success")
    return redirect(url_for("appraisal.admin_manage_criteria"))

# -----------------------------
# Employee: Self-appraisal
# -----------------------------
@appraisal_bp.route("/employee/new", methods=["GET", "POST"])
@login_required
def employee_new_appraisal():
    # role guard
    if _role_upper() != "EMPLOYEE":
        flash("Access denied.", "danger")
        return redirect(url_for("home", open="login_employee"))

    # criteria (stable order)
    criteria = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    criteria_by_id = {c.id: f"[{c.type}] {c.name}" for c in criteria}

    form = AppraisalForm()

    # Prefill read-only staff info
    form.employee_name.data   = getattr(current_user, "full_name", None) or current_user.username
    form.hod_name.data        = getattr(current_user, "hod_name", "")
    form.position_title.data  = getattr(current_user, "position_title", "")
    form.department_name.data = getattr(current_user, "department_name", "")
    profile = getattr(current_user, "staff_profile", None)
    if profile and getattr(profile, "entry_date", None):
        form.entry_date.data = profile.entry_date

    # Seed teaching rows
    while len(form.teach_s2) < 4:
        form.teach_s2.append_entry()
    while len(form.teach_s1) < 3:
        form.teach_s1.append_entry()

    # Seed personal skills label rows (supervisor rates later)
    skill_labels = [
        "Has appropriate knowledge for position",
        "Analyses problems and seeks solutions",
        "Is committed, reliable and dependable",
        "Uses ICT/workplace equipment competently",
        "Communicates with clear oral and written expression",
        "Works productively, autonomously and co-operatively",
        "Has regular attendance, is punctual, completes tasks on time",
        "Has effective planning and organization skills",
        "Shows initiative",
        "Is friendly, polite and patient in dealing with others",
    ]
    while len(form.skill_ratings) < len(skill_labels):
        form.skill_ratings.append_entry()
    for i, lbl in enumerate(skill_labels):
        form.skill_ratings[i].label_text.data = lbl

    # Build duty/skill rows (employee provides examples only)
    try:
        form.scores.entries = []
    except Exception:
        pass
    for c in criteria:
        form.scores.append_entry({
            "criteria_id": c.id,
            "rating": 3,        # placeholder (manager will overwrite)
            "example_text": "",
        })

    # Headings year
    form.eval_year.data = (form.period_end.data.year if form.period_end.data else date.today().year)

    # POST: manual validation + save
    if request.method == "POST":
        errors = []
        if not form.period_start.data or not form.period_end.data:
            errors.append("Please provide the evaluation period (start and end).")

        # time split check if provided
        t = float(form.teaching_percent.data or 0)
        r = float(form.research_percent.data or 0)
        o = float(form.other_percent.data or 0)
        total = round(t + r + o, 1)
        if total != 0 and abs(total - 100.0) > 0.1:
            errors.append("Time allocation must total 100%.")

        if errors:
            for e in errors:
                flash(e, "warning")
            return render_template(
                "appraisal/appraisal_form.html",
                form=form,
                role="employee",
                criteria_by_id=criteria_by_id,
            )

        # Find manager (CycleEnrollment preferred)
        enroll = (CycleEnrollment.query
                  .filter_by(employee_id=current_user.id)
                  .order_by(CycleEnrollment.id.desc())
                  .first())
        manager_id = (enroll.manager_id if enroll and enroll.manager_id
                      else getattr(current_user, "manager_id", None) or 1)

        # Create appraisal shell
        appraisal = Appraisal(
            employee_id=current_user.id,
            supervisor_id=manager_id,
            period_start=form.period_start.data,
            period_end=form.period_end.data,
            employee_comment=form.employee_comment.data,
            status=ReviewStatus.SUBMITTED_TO_MANAGER,
        )
        # Save publications + teaching workload JSON
        appraisal.publications_text = form.publications_2020_2024.data or ""

        def _rows(fieldlist):
            rows = []
            for r in fieldlist.entries:
                sub = r.form
                row = {
                    "unit_code":      (sub.unit_code.data or "").strip(),
                    "unit_name":      (sub.unit_name.data or "").strip(),
                    "group_name":     (sub.group_name.data or "").strip(),
                    "num_students":   sub.num_students.data if sub.num_students.data is not None else None,
                    "session_type":   (sub.session_type.data or "").strip(),
                    "hours_per_week": float(sub.hours_per_week.data) if sub.hours_per_week.data not in (None, "") else None,
                }
                if any([row["unit_code"], row["unit_name"], row["group_name"],
                        row["num_students"] not in (None, ""), row["session_type"],
                        row["hours_per_week"] not in (None, "")]):
                    rows.append(row)
            return rows

        appraisal.teach_s2_json = _rows(form.teach_s2)
        appraisal.teach_s1_json = _rows(form.teach_s1)

        db.session.add(appraisal)
        db.session.commit()  # get id

        # Save workload split
        db.session.add(Workload(
            appraisal_id=appraisal.id,
            teaching_percent=form.teaching_percent.data or 0,
            research_percent=form.research_percent.data or 0,
            other_percent=form.other_percent.data or 0,
        ))

        # Save employee examples (ratings temp = 3)
        for idx, c in enumerate(criteria):
            example_txt = form.scores[idx].example_text.data if idx < len(form.scores) else ""
            db.session.add(Score(
                appraisal_id=appraisal.id,
                criteria_id=c.id,
                rating=3,
                example_text=example_txt,
            ))

        db.session.commit()
        flash("✅ Appraisal submitted to your manager.", "success")
        return redirect(url_for("appraisal.employee_list_appraisals"))

    # GET
    return render_template(
        "appraisal/appraisal_form.html",
        form=form,
        role="employee",
        criteria_by_id=criteria_by_id,
    )


@appraisal_bp.route("/employee/list")
@login_required
def employee_list_appraisals():
    appraisals = (Appraisal.query
                  .filter_by(employee_id=current_user.id)
                  .order_by(Appraisal.created_at.desc())
                  .all())
    return render_template("appraisal/appraisal_list.html",
                           appraisals=appraisals, role="employee")

# -----------------------------
# Manager views
# -----------------------------
@appraisal_bp.route("/manager/list")
@login_required
def manager_list():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisals = (Appraisal.query
                  .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
                  .filter(or_(Appraisal.supervisor_id == current_user.id,
                              CycleEnrollment.manager_id == current_user.id))
                  .order_by(Appraisal.created_at.desc())
                  .all())
    _attach_people(appraisals)
    return render_template("appraisal/manager_list.html", appraisals=appraisals, role="manager")


# Manager: Review a single appraisal (full mirror of employee form)
@appraisal_bp.route("/manager/review/<int:appraisal_id>", methods=["GET", "POST"], endpoint="manager_review")
@login_required
def manager_review(appraisal_id):
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisal = Appraisal.query.get_or_404(appraisal_id)

    # allow if this manager is the supervisor on the record OR mapped via cycle
    mapped = CycleEnrollment.query.filter_by(
        employee_id=appraisal.employee_id, manager_id=current_user.id
    ).first() is not None
    if not (appraisal.supervisor_id == current_user.id or mapped):
        flash("Not your appraisal to review.", "danger")
        return redirect(url_for("appraisal.manager_list"))

    employee = User.query.get(appraisal.employee_id)
    supervisor_user = User.query.get(appraisal.supervisor_id)
    wl = Workload.query.filter_by(appraisal_id=appraisal.id).first()
    criteria = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    criteria_by_id = {c.id: f"[{c.type}] {c.name}" for c in criteria}

    def _pick(*vals):
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    form = AppraisalForm()

    # A. staff info (read-only in template)
    form.employee_name.data   = _pick(getattr(employee, "full_name", None),
                                      getattr(employee, "username", None),
                                      f"Employee #{appraisal.employee_id}")
    form.hod_name.data        = _pick(getattr(supervisor_user, "full_name", None),
                                      getattr(supervisor_user, "username", None),
                                      "Head of Department")
    form.position_title.data  = _pick(getattr(employee, "position_title", None),
                                      getattr(employee, "job_title", None),
                                      (getattr(getattr(employee, "role", None), "value", None) or "").title(),
                                      "Staff")
    form.department_name.data = _pick(getattr(employee, "department_name", None),
                                      getattr(employee, "department", None),
                                      "Department")
    prof = getattr(employee, "staff_profile", None)
    entry_dt = (getattr(prof, "entry_date", None) or
                getattr(employee, "date_joined", None) or
                getattr(employee, "created_at", None))
    if entry_dt:
        form.entry_date.data = entry_dt

    # B. period + heading year
    form.period_start.data = appraisal.period_start
    form.period_end.data   = appraisal.period_end
    form.eval_year.data = (
        appraisal.period_end.year if appraisal.period_end else
        (appraisal.period_start.year if appraisal.period_start else date.today().year)
    )

    # C. workload % (employee data)
    if wl:
        form.teaching_percent.data = wl.teaching_percent or 0
        form.research_percent.data = wl.research_percent or 0
        form.other_percent.data    = wl.other_percent or 0

    # D. publications
    form.publications_2020_2024.data = getattr(appraisal, "publications_text", "") or ""

    # E/F. teaching workload rows (hydrate from JSON)
    def _ensure_rows(fieldlist, rows, min_rows):
        while len(fieldlist) < max(len(rows), min_rows):
            fieldlist.append_entry()
        for i, row in enumerate(rows[:len(fieldlist)]):
            sub = fieldlist[i].form
            sub.unit_code.data      = row.get("unit_code") or ""
            sub.unit_name.data      = row.get("unit_name") or ""
            sub.group_name.data     = row.get("group_name") or ""
            sub.num_students.data   = row.get("num_students")
            sub.session_type.data   = row.get("session_type") or ""
            sub.hours_per_week.data = row.get("hours_per_week")

    _ensure_rows(form.teach_s2, appraisal.teach_s2_json or [], 4)
    _ensure_rows(form.teach_s1, appraisal.teach_s1_json or [], 3)

    # G. duties/skills: build with existing values
    try:
        form.scores.entries = []
    except Exception:
        pass
    for c in criteria:
        existing = next((s for s in appraisal.scores if s.criteria_id == c.id), None)
        form.scores.append_entry({
            "criteria_id": c.id,
            "rating": (existing.rating if existing else 3),
            "example_text": (existing.example_text if existing else ""),
        })

    # I/J. comments/signatures
    form.employee_comment.data    = appraisal.employee_comment or ""
    form.supervisor_comments.data = appraisal.supervisor_comment or ""

    # POST: save + submit to HR
    if request.method == "POST":
        appraisal.supervisor_comment = (request.form.get(form.supervisor_comments.name) or "").strip()

        total = 0
        for idx, c in enumerate(criteria):
            rating_raw = request.form.get(f"scores-{idx}-rating")
            example    = request.form.get(f"scores-{idx}-example_text")
            try:
                rating = int(rating_raw)
            except Exception:
                rating = 3

            score = Score.query.filter_by(appraisal_id=appraisal.id, criteria_id=c.id).first()
            if score:
                score.rating = rating
                score.example_text = example
            else:
                db.session.add(Score(
                    appraisal_id=appraisal.id,
                    criteria_id=c.id,
                    rating=rating,
                    example_text=example,
                ))
            total += rating

        appraisal.total_score = total
        appraisal.status = ReviewStatus.MANAGER_SUBMITTED
        db.session.commit()
        flash("✅ Submitted to HR.", "success")
        return redirect(url_for("appraisal.manager_list"))

    # Render manager mirror form
    return render_template(
        "appraisal/appraisal_form_manager.html",
        form=form,
        appraisal=appraisal,
        employee=employee,
        criteria_by_id=criteria_by_id,
        role="manager",
        quickfill_url=url_for("appraisal.manager_quickfill", appraisal_id=appraisal.id),
    )

# --- NEW: Quick Fill & Submit (demo helper) -------------------
@appraisal_bp.route("/manager/review/<int:appraisal_id>/quickfill", methods=["POST"], endpoint="manager_quickfill")
@login_required
def manager_quickfill(appraisal_id):
    """Auto-fill ratings/comments and submit to HR. Use for demos/testing."""
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisal = Appraisal.query.get_or_404(appraisal_id)
    mapped = CycleEnrollment.query.filter_by(
        employee_id=appraisal.employee_id, manager_id=current_user.id
    ).first() is not None
    if not (appraisal.supervisor_id == current_user.id or mapped):
        flash("Not your appraisal to review.", "danger")
        return redirect(url_for("appraisal.manager_list"))

    criteria = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    total = 0
    for c in criteria:
        score = Score.query.filter_by(appraisal_id=appraisal.id, criteria_id=c.id).first()
        if not score:
            score = Score(appraisal_id=appraisal.id, criteria_id=c.id)
            db.session.add(score)
        score.rating = 4  # default demo rating
        if not score.example_text:
            score.example_text = "See employee example above."
        total += score.rating or 0

    if not appraisal.supervisor_comment:
        appraisal.supervisor_comment = "Reviewed. Meets expectations across the criteria. Proceed."

    appraisal.total_score = total
    appraisal.status = ReviewStatus.MANAGER_SUBMITTED
    db.session.commit()
    flash("✅ Filled with sample ratings and submitted to HR.", "success")
    return redirect(url_for("appraisal.manager_list"))

@appraisal_bp.route("/manager/pending")
@login_required
def manager_pending():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    rows = (Appraisal.query
            .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
            .filter(
                Appraisal.status == ReviewStatus.SUBMITTED_TO_MANAGER,
                or_(Appraisal.supervisor_id == current_user.id,
                    CycleEnrollment.manager_id == current_user.id)
            )
            .order_by(Appraisal.created_at.desc())
            .all())
    _attach_people(rows)
    return render_template("appraisal/manager_pending.html", appraisals=rows, page_title="Pending Appraisals")

@appraisal_bp.route("/manager/completed")
@login_required
def manager_completed():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    rows = (Appraisal.query
            .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
            .filter(
                or_(Appraisal.supervisor_id == current_user.id,
                    CycleEnrollment.manager_id == current_user.id),
                Appraisal.status.in_([ReviewStatus.HR_REVIEWED, ReviewStatus.APPROVED])
            )
            .order_by(Appraisal.created_at.desc())
            .all())
    _attach_people(rows)
    return render_template("appraisal/manager_completed.html", appraisals=rows)

@appraisal_bp.route("/employee/view/<int:appraisal_id>", methods=["GET"], endpoint="employee_view")
@login_required
def employee_view(appraisal_id):
    # Only the owner (employee) can view
    if _role_upper() != "EMPLOYEE":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisal = Appraisal.query.get_or_404(appraisal_id)
    if appraisal.employee_id != current_user.id:
        flash("Not your appraisal.", "danger")
        return redirect(url_for("appraisal.employee_list_appraisals"))

    # Build a read-only view using the same form data as manager_review
    employee = User.query.get(appraisal.employee_id)
    supervisor_user = User.query.get(appraisal.supervisor_id)
    wl = Workload.query.filter_by(appraisal_id=appraisal.id).first()
    criteria = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    criteria_by_id = {c.id: f"[{c.type}] {c.name}" for c in criteria}

    def _pick(*vals):
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    form = AppraisalForm()
    # Staff info
    form.employee_name.data   = _pick(getattr(employee, "full_name", None),
                                      getattr(employee, "username", None),
                                      f"Employee #{appraisal.employee_id}")
    form.hod_name.data        = _pick(getattr(supervisor_user, "full_name", None),
                                      getattr(supervisor_user, "username", None),
                                      "Head of Department")
    form.position_title.data  = _pick(getattr(employee, "position_title", None),
                                      getattr(employee, "job_title", None),
                                      (getattr(getattr(employee, "role", None), "value", None) or "").title(),
                                      "Staff")
    form.department_name.data = _pick(getattr(employee, "department_name", None),
                                      getattr(employee, "department", None),
                                      "Department")
    prof = getattr(employee, "staff_profile", None)
    entry_dt = (getattr(prof, "entry_date", None) or
                getattr(employee, "date_joined", None) or
                getattr(employee, "created_at", None))
    if entry_dt:
        form.entry_date.data = entry_dt

    # Period/year
    form.period_start.data = appraisal.period_start
    form.period_end.data   = appraisal.period_end
    form.eval_year.data = (appraisal.period_end.year if appraisal.period_end
                           else (appraisal.period_start.year if appraisal.period_start else date.today().year))

    # Workload %
    if wl:
        form.teaching_percent.data = wl.teaching_percent or 0
        form.research_percent.data = wl.research_percent or 0
        form.other_percent.data    = wl.other_percent or 0

    # Publications
    form.publications_2020_2024.data = getattr(appraisal, "publications_text", "") or ""

    # Teaching rows (hydrate)
    def _ensure_rows(fieldlist, rows, min_rows):
        while len(fieldlist) < max(len(rows), min_rows):
            fieldlist.append_entry()
        for i, row in enumerate(rows[:len(fieldlist)]):
            sub = fieldlist[i].form
            sub.unit_code.data      = row.get("unit_code") or ""
            sub.unit_name.data      = row.get("unit_name") or ""
            sub.group_name.data     = row.get("group_name") or ""
            sub.num_students.data   = row.get("num_students")
            sub.session_type.data   = row.get("session_type") or ""
            sub.hours_per_week.data = row.get("hours_per_week")

    _ensure_rows(form.teach_s2, appraisal.teach_s2_json or [], 4)
    _ensure_rows(form.teach_s1, appraisal.teach_s1_json or [], 3)

    # Criteria/scores
    try:
        form.scores.entries = []
    except Exception:
        pass
    for c in criteria:
        existing = next((s for s in appraisal.scores if s.criteria_id == c.id), None)
        form.scores.append_entry({
            "criteria_id": c.id,
            "rating": (existing.rating if existing else 3),
            "example_text": (existing.example_text if existing else ""),
        })

    # Comments
    form.employee_comment.data    = appraisal.employee_comment or ""
    form.supervisor_comments.data = appraisal.supervisor_comment or ""

    # Render as read-only (your template can use `readonly` to disable inputs)
    return render_template(
        "appraisal/appraisal_form.html",
        form=form,
        appraisal=appraisal,
        employee=employee,
        criteria_by_id=criteria_by_id,
        role="employee_view",
        readonly=True,
    )

# --- Delete appraisal (Employee/Manager/HR) --------------------
from flask import abort

# --- Delete appraisal (Employee / Manager / HR) -----------------
@appraisal_bp.route("/delete/<int:appraisal_id>", methods=["POST"], endpoint="delete_appraisal")
@login_required
def delete_appraisal(appraisal_id):
    a = Appraisal.query.get_or_404(appraisal_id)
    role = _role_upper()

    # Who can delete?
    is_hr_or_admin = role in {"HR", "ADMIN"}

    # Employees: only their own and only while not finalized
    allowed_employee_statuses = set()
    for name in ("DRAFT", "SUBMITTED_TO_MANAGER"):
        if hasattr(ReviewStatus, name):
            allowed_employee_statuses.add(getattr(ReviewStatus, name))

    is_owner_and_allowed = (
        role == "EMPLOYEE"
        and a.employee_id == current_user.id
        and (a.status in allowed_employee_statuses)
    )

    # Managers: if they supervise directly or via cycle mapping
    mapped = CycleEnrollment.query.filter_by(
        employee_id=a.employee_id, manager_id=current_user.id
    ).first() is not None
    is_manager_for_item = (role == "MANAGER" and (a.supervisor_id == current_user.id or mapped))

    if not (is_hr_or_admin or is_owner_and_allowed or is_manager_for_item):
        flash("You don't have permission to delete this appraisal.", "danger")
        if role == "EMPLOYEE":
            return redirect(url_for("appraisal.employee_list_appraisals"))
        if role == "MANAGER":
            return redirect(url_for("appraisal.manager_list"))
        return redirect(url_for("dashboard"))

    # Remove children if cascade isn't configured on relationships
    Score.query.filter_by(appraisal_id=appraisal_id).delete(synchronize_session=False)
    Workload.query.filter_by(appraisal_id=appraisal_id).delete(synchronize_session=False)

    db.session.delete(a)
    db.session.commit()
    flash(f"Appraisal #{appraisal_id} deleted.", "success")

    # Redirect to the right list
    if role == "EMPLOYEE":
        return redirect(url_for("appraisal.employee_list_appraisals"))
    if role == "MANAGER":
        return redirect(url_for("appraisal.manager_list"))
    return redirect(url_for("appraisal.hr_reports"))


# --- Compatibility shim: generic detail redirect --------------------------------
@appraisal_bp.route("/detail/<int:appraisal_id>", methods=["GET"])
@login_required
def appraisal_detail(appraisal_id):
    role = _role_upper()
    if role == "HR":
        return redirect(url_for("appraisal.hr_review_detail", appraisal_id=appraisal_id))
    if role == "MANAGER":
        return redirect(url_for("appraisal.manager_review", appraisal_id=appraisal_id))
    if role == "EMPLOYEE":
        return redirect(url_for("appraisal.employee_view", appraisal_id=appraisal_id))
    flash("Access denied.", "danger")
    return redirect(url_for("dashboard"))



# -----------------------------
# Manager Exports
# -----------------------------
@appraisal_bp.route("/manager/reports/pdf")
@login_required
def manager_reports_pdf():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisals = Appraisal.query.filter_by(supervisor_id=current_user.id).order_by(Appraisal.created_at.desc()).all()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    p.setFont("Helvetica-Bold", 14)
    p.drawString(180, y, "Manager Appraisal Report")
    y -= 30

    p.setFont("Helvetica", 10)
    for a in appraisals:
        st = getattr(a.status, "value", str(a.status))
        text = f"ID: {a.id} | Emp: {a.employee_id} | {a.period_start}→{a.period_end} | Score: {a.total_score or 'Pending'} | {st}"
        p.drawString(50, y, text)
        y -= 20
        if y < 50:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 10)

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="manager_appraisals.pdf",
                     mimetype="application/pdf")

@appraisal_bp.route("/manager/submitted")
@login_required
def manager_submitted():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    rows = (Appraisal.query
            .outerjoin(CycleEnrollment, CycleEnrollment.employee_id == Appraisal.employee_id)
            .filter(
                Appraisal.status == ReviewStatus.MANAGER_SUBMITTED,
                or_(Appraisal.supervisor_id == current_user.id,
                    CycleEnrollment.manager_id == current_user.id)
            )
            .order_by(Appraisal.created_at.desc())
            .all())
    _attach_people(rows)

    return render_template("appraisal/manager_pending.html",
                           appraisals=rows,
                           page_title="Awaiting HR (Submitted)")

@appraisal_bp.route("/manager/reports/excel")
@login_required
def manager_reports_excel():
    if _role_upper() != "MANAGER":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisals = Appraisal.query.filter_by(supervisor_id=current_user.id).order_by(Appraisal.created_at.desc()).all()
    data = [
        {
            "ID": a.id,
            "Employee ID": a.employee_id,
            "Period Start": a.period_start,
            "Period End": a.period_end,
            "Total Score": a.total_score,
            "Status": getattr(a.status, "value", str(a.status)),
        }
        for a in appraisals
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, index=False, sheet_name="My Team Appraisals")

    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="manager_appraisals.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------
# HR Review Queue & Actions
# -----------------------------
@appraisal_bp.route("/hr/reports")
@login_required
def hr_reports():
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    rows = Appraisal.query.order_by(Appraisal.created_at.desc()).all()
    _attach_people(rows)
    return render_template("appraisal/hr_reports.html", appraisals=rows, role="hr")

@appraisal_bp.route("/hr/reviews")
@login_required
def hr_review_queue_ui():
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    rows = Appraisal.query.filter_by(status=ReviewStatus.MANAGER_SUBMITTED).order_by(Appraisal.created_at.desc()).all()
    _attach_people(rows)
    return render_template("appraisal/hr_reports.html", appraisals=rows, role="hr")

@appraisal_bp.route("/hr/approve/<int:appraisal_id>", methods=["POST"])
@login_required
def hr_approve(appraisal_id):
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    a = Appraisal.query.get_or_404(appraisal_id)
    a.status = ReviewStatus.APPROVED
    db.session.commit()
    flash("Appraisal approved.", "success")
    return redirect(url_for("appraisal.hr_review_queue_ui"))

@appraisal_bp.route("/hr/reject/<int:appraisal_id>", methods=["POST"])
@login_required
def hr_reject(appraisal_id):
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    a = Appraisal.query.get_or_404(appraisal_id)
    a.status = ReviewStatus.REJECTED
    db.session.commit()
    flash("Appraisal rejected.", "danger")
    return redirect(url_for("appraisal.hr_review_queue_ui"))

# -----------------------------
# HR Exports
# -----------------------------
@appraisal_bp.route("/hr/export/pdf")
@login_required
def hr_export_pdf():
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    appraisals = Appraisal.query.order_by(Appraisal.created_at.desc()).all()
    y = height - 50
    p.setFont("Helvetica-Bold", 14)
    p.drawString(200, y, "HR Appraisal Report")
    y -= 30

    p.setFont("Helvetica", 10)
    for a in appraisals:
        st = getattr(a.status, "value", str(a.status))
        text = f"ID: {a.id} | Emp: {a.employee_id} | Sup: {a.supervisor_id} | {a.period_start}→{a.period_end} | Score: {a.total_score or 'Pending'} | {st}"
        p.drawString(50, y, text)
        y -= 20
        if y < 50:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 10)

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="hr_appraisals.pdf", mimetype="application/pdf")

@appraisal_bp.route("/hr/export/excel")
@login_required
def hr_export_excel():
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisals = Appraisal.query.order_by(Appraisal.created_at.desc()).all()
    data = [
        {
            "ID": a.id,
            "Employee ID": a.employee_id,
            "Supervisor ID": a.supervisor_id,
            "Period Start": a.period_start,
            "Period End": a.period_end,
            "Total Score": a.total_score,
            "Status": getattr(a.status, "value", str(a.status)),
        }
        for a in appraisals
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, index=False, sheet_name="Appraisals")

    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="hr_appraisals.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@appraisal_bp.route("/hr/export/csv")
@login_required
def hr_export_csv():
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    appraisals = Appraisal.query.order_by(Appraisal.created_at.desc()).all()
    data = [
        {
            "ID": a.id,
            "Employee ID": a.employee_id,
            "Supervisor ID": a.supervisor_id,
            "Period Start": a.period_start,
            "Period End": a.period_end,
            "Total Score": a.total_score,
            "Status": getattr(a.status, "value", str(a.status)),
        }
        for a in appraisals
    ]
    csv_bytes = pd.DataFrame(data).to_csv(index=False).encode("utf-8")
    return send_file(io.BytesIO(csv_bytes), as_attachment=True,
                     download_name="hr_appraisals.csv", mimetype="text/csv")

# -----------------------------
# HR: Criteria Library (UI)
# -----------------------------
@appraisal_bp.route("/criteria/ui", methods=["GET", "POST"], endpoint="criteria_ui")
@login_required
@hr_required
def criteria_ui():
    if request.method == "POST":
        action = (request.form.get("action") or "delete").lower()

        if action == "add":
            name = (request.form.get("name") or "").strip()
            type_ = (request.form.get("type") or "").strip()
            if not name or not type_:
                flash("Please provide both Type and Name.", "danger")
            else:
                db.session.add(Criteria(name=name, type=type_))
                db.session.commit()
                flash("Criteria added.", "success")
            return redirect(url_for("appraisal.criteria_ui"))

        if action == "delete":
            crit_id = request.form.get("criteria_id")
            try:
                cid = int(crit_id)
            except (TypeError, ValueError):
                flash("Invalid criterion id.", "danger")
                return redirect(url_for("appraisal.criteria_ui"))

            in_use = Score.query.filter_by(criteria_id=cid).limit(1).first() is not None
            if in_use:
                flash("Cannot delete: criterion is used in one or more appraisals.", "warning")
                return redirect(url_for("appraisal.criteria_ui"))

            obj = Criteria.query.get(cid)
            if not obj:
                flash("Criterion not found.", "danger")
            else:
                db.session.delete(obj)
                db.session.commit()
                flash("Criteria deleted.", "success")
            return redirect(url_for("appraisal.criteria_ui"))

    items = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    return render_template("appraisal/criteria_ui.html",
                           title="Define Criteria",
                           criteria=items)


@appraisal_bp.route("/criteria", methods=["GET"])
@login_required
@hr_required
def criteria_redirect():
    return redirect(url_for("appraisal.criteria_ui"))

# Simple placeholders used by your dashboards
@appraisal_bp.route("/cycle/new/ui")
@login_required
def cycle_new_ui():
    return render_template("hr_placeholder.html", title="Launch Cycle")

@appraisal_bp.route("/reviewers/ui")
@login_required
def assign_reviewers_ui():
    return render_template("hr_placeholder.html", title="Assign Reviewers")

@appraisal_bp.route("/progress/ui")
@login_required
def progress_ui():
    return render_template("hr_placeholder.html", title="Monitor Progress")

@appraisal_bp.route("/reviews/ui")
@login_required
def hr_reviews_ui_alias():
    return redirect(url_for("appraisal.hr_review_queue_ui"))

@appraisal_bp.route("/calibration/ui")
@login_required
def calibrate_ui():
    return render_template("hr_placeholder.html", title="Calibrate Ratings")

@appraisal_bp.route("/communicate/ui")
@login_required
def communicate_ui():
    return render_template("hr_placeholder.html", title="Communicate Outcomes")

# =============================
# HR: Detailed review page (OPEN)
# =============================
@appraisal_bp.route("/hr/review/<int:appraisal_id>", methods=["GET", "POST"], endpoint="hr_review_detail")
@login_required
def hr_review_detail(appraisal_id):
    if _role_upper() != "HR":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    a = Appraisal.query.get_or_404(appraisal_id)
    employee   = User.query.get(a.employee_id)
    supervisor = User.query.get(a.supervisor_id)
    wl = Workload.query.filter_by(appraisal_id=a.id).first()

    # Pull criteria & scores in stable order
    criteria = Criteria.query.order_by(Criteria.type.asc(), Criteria.id.asc()).all()
    score_map = {s.criteria_id: s for s in a.scores}

    rows, total, cnt = [], 0, 0
    dist = {1:0, 2:0, 3:0, 4:0, 5:0}
    for c in criteria:
        sc = score_map.get(c.id)
        rating = sc.rating if sc else None
        example = sc.example_text if sc else ""
        rows.append({
            "label": f"[{c.type}] {c.name}",
            "rating": rating,
            "example": example
        })
        if isinstance(rating, int):
            total += rating
            cnt += 1
            if rating in dist: dist[rating] += 1

    avg = round(total / cnt, 2) if cnt else None

    # Handle Approve/Reject POST
    if request.method == "POST":
        action = (request.form.get("action") or "").lower()
        if action == "approve":
            a.status = ReviewStatus.APPROVED
            db.session.commit()
            flash("✅ Appraisal approved.", "success")
        elif action == "reject":
            a.status = ReviewStatus.REJECTED
            db.session.commit()
            flash("⛔ Appraisal rejected.", "danger")
        else:
            flash("Unknown action.", "warning")
        return redirect(url_for("appraisal.hr_review_queue_ui"))

    return render_template(
        "appraisal/hr_review_detail.html",
        a=a,
        employee=employee,
        supervisor=supervisor,
        wl=wl,
        rows=rows,
        avg=avg,
        dist=dist,
        cnt=cnt
    )
