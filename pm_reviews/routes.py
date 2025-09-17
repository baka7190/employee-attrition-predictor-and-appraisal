# pm_reviews/routes.py
from functools import wraps
from datetime import date
from flask import render_template, request, redirect, url_for, flash, abort, session

from . import bp
from extensions import db
from models import (
    User,
    PerformanceReview,
    ReviewStatus,
    ReviewAudit,
    Role,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _role_str(r):
    """Return enum/string as canonical role string ('EMPLOYEE'|'MANAGER'|'HR')."""
    return r.value if isinstance(r, Role) else str(r).upper()


def _portal_for(role_str):
    """Return a redirect to the proper login modal for the given role string."""
    role_str = _role_str(role_str)
    target = "login_employee"
    if role_str == "MANAGER":
        target = "login_manager"
    elif role_str == "HR":
        target = "login_hr"
    return redirect(url_for("home", open=target))


def _session_role():
    """Role string from session; if missing, load from DB and cache into session."""
    role = session.get("role")
    if role:
        return role
    uid = session.get("user_id")
    if not uid:
        return None
    u = User.query.get(uid)
    if u and u.role:
        session["role"] = u.role.value
        return u.role.value
    return None


def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None


def can_view_review(review: PerformanceReview, user: User) -> bool:
    if user.role == Role.HR:
        return True
    if user.role == Role.MANAGER and review.manager_id == user.id:
        return True
    return review.employee_id == user.id


# ---------------------------------------------------------------------------
# decorators
# ---------------------------------------------------------------------------

def login_required_session(expected_role: Role | str | None = None):
    """
    Require login. If not logged in, open the appropriate login modal.
    If expected_role is set and the logged user has a different role,
    redirect to that role's portal.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return _portal_for(expected_role or Role.EMPLOYEE)
            if expected_role:
                current = _session_role()
                if current != _role_str(expected_role):
                    return _portal_for(expected_role)
            return fn(*args, **kwargs)
        return wrapper
    return deco


def require_roles_session(*roles):
    """
    Require that the logged-in user has one of the given roles.
    If not logged in, open portal for the first role.
    If logged with wrong role, open portal for the first role.
    """
    allowed = {_role_str(r) for r in roles}
    primary = next(iter(allowed)) if allowed else "EMPLOYEE"

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return _portal_for(primary)
            current = _session_role()
            if current not in allowed:
                return _portal_for(primary)
            return fn(*args, **kwargs)
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# EMPLOYEE
# ---------------------------------------------------------------------------

@bp.route("/new", methods=["GET", "POST"])
@login_required_session(Role.EMPLOYEE)
def new_review():
    u = current_user()
    if request.method == "POST":
        r = PerformanceReview(
            employee_id=u.id,
            manager_id=u.manager_id,
            period_start=date.fromisoformat(request.form["period_start"]),
            period_end=date.fromisoformat(request.form["period_end"]),
            self_goals=request.form.get("self_goals", ""),
            self_achievements=request.form.get("self_achievements", ""),
            self_training=request.form.get("self_training", ""),
            self_rating=int(request.form.get("self_rating") or 0),
        )
        db.session.add(r)
        db.session.commit()
        flash("Draft created.", "success")
        return redirect(url_for("reviews.self_review", review_id=r.id))
    return render_template("reviews/new.html")


@bp.route("/<int:review_id>/self", methods=["GET", "POST"])
@login_required_session(Role.EMPLOYEE)
def self_review(review_id):
    """Employee edits own review; submits to manager."""
    u = current_user()
    review = PerformanceReview.query.get_or_404(review_id)

    # Only owner can edit; allow editing when in DRAFT
    if review.employee_id != u.id or review.status not in {ReviewStatus.DRAFT}:
        abort(403)

    if request.method == "POST":
        review.self_goals = request.form.get("goals", request.form.get("self_goals", ""))
        review.self_achievements = request.form.get("achievements", request.form.get("self_achievements", ""))
        review.self_training = request.form.get("training", request.form.get("self_training", ""))
        review.self_rating = int(request.form.get("self_rating") or 0)

        action = (request.form.get("action") or "").lower()
        if action == "submit" or "submit_to_manager" in request.form:
            prev = review.status
            review.status = ReviewStatus.SUBMITTED_TO_MANAGER
            db.session.add(
                ReviewAudit(
                    review_id=review.id,
                    actor_id=u.id,
                    from_status=prev,
                    to_status=review.status,
                    note="Employee submitted to manager",
                )
            )
            flash("Submitted to your manager.", "success")
        else:
            flash("Draft saved.", "success")

        db.session.commit()
        return redirect(url_for("reviews.employee_outbox"))

    # GET
    return render_template("reviews/self.html", review=review)


@bp.route("/me/outbox")
@login_required_session(Role.EMPLOYEE)
def employee_outbox():
    u = current_user()
    reviews = (
        PerformanceReview.query.filter_by(employee_id=u.id)
        .order_by(PerformanceReview.updated_at.desc())
        .all()
    )
    return render_template("reviews/outbox_employee.html", reviews=reviews)


# View-only page for any allowed user (employee/manager/HR)
@bp.route("/<int:review_id>/view", endpoint="view_review")
@login_required_session()  # just requires login; can_view_review enforces access
def view_review(review_id):
    u = current_user()
    review = PerformanceReview.query.get_or_404(review_id)
    if not can_view_review(review, u):
        abort(403)
    return render_template("reviews/view.html", review=review)


# ---------------------------------------------------------------------------
# MANAGER
# ---------------------------------------------------------------------------

@bp.route("/manager/inbox")
@require_roles_session(Role.MANAGER)
def manager_inbox():
    u = current_user()
    reviews = PerformanceReview.query.filter_by(
        manager_id=u.id, status=ReviewStatus.SUBMITTED_TO_MANAGER
    ).all()
    return render_template("reviews/manager_inbox.html", reviews=reviews)


@bp.route("/<int:review_id>/manager", methods=["GET", "POST"])
@require_roles_session(Role.MANAGER)
def manager_review(review_id):
    u = current_user()
    review = PerformanceReview.query.get_or_404(review_id)
    if review.manager_id != u.id or review.status != ReviewStatus.SUBMITTED_TO_MANAGER:
        abort(403)

    if request.method == "POST":
        review.manager_comments = request.form.get("manager_comments", "")
        review.manager_rating = int(request.form.get("manager_rating") or 0)

        prev = review.status
        review.status = ReviewStatus.MANAGER_SUBMITTED
        db.session.add(
            ReviewAudit(
                review_id=review.id,
                actor_id=u.id,
                from_status=prev,
                to_status=review.status,
                note="Manager submitted to HR",
            )
        )
        db.session.commit()
        flash("Forwarded to HR.", "success")
        return redirect(url_for("reviews.manager_inbox"))

    return render_template("reviews/manager_review.html", review=review)


# ---------------------------------------------------------------------------
# HR
# ---------------------------------------------------------------------------

@bp.route("/hr/inbox")
@require_roles_session(Role.HR)
def hr_inbox():
    waiting = PerformanceReview.query.filter_by(status=ReviewStatus.MANAGER_SUBMITTED).all()
    decided = PerformanceReview.query.filter(
        PerformanceReview.status.in_(
            [ReviewStatus.HR_REVIEWED, ReviewStatus.APPROVED, ReviewStatus.REJECTED]
        )
    ).all()
    return render_template("reviews/hr_inbox.html", waiting=waiting, decided=decided)


@bp.route("/<int:review_id>/hr", methods=["GET", "POST"])
@require_roles_session(Role.HR)
def hr_review(review_id):
    u = current_user()
    review = PerformanceReview.query.get_or_404(review_id)
    if review.status != ReviewStatus.MANAGER_SUBMITTED:
        abort(403)

    if request.method == "POST":
        review.hr_comments = request.form.get("hr_comments", "")
        review.promotion_recommended = bool(request.form.get("promotion_recommended"))
        review.new_title = (request.form.get("new_title") or None) or None

        new_salary_raw = request.form.get("new_salary")
        review.new_salary = float(new_salary_raw) if new_salary_raw else None

        eff = request.form.get("effective_date")
        review.effective_date = date.fromisoformat(eff) if eff else None

        # decision: approve / reject / reviewed
        decision = (request.form.get("decision") or "").lower()
        prev = review.status
        if decision == "approve" or review.promotion_recommended:
            review.status = ReviewStatus.APPROVED
            note = "HR approved"
        elif decision == "reject":
            review.status = ReviewStatus.REJECTED
            note = "HR rejected"
        else:
            review.status = ReviewStatus.HR_REVIEWED
            note = "HR reviewed"

        db.session.add(
            ReviewAudit(
                review_id=review.id,
                actor_id=u.id,
                from_status=prev,
                to_status=review.status,
                note=note,
            )
        )
        db.session.commit()
        flash("HR decision recorded.", "success")
        return redirect(url_for("reviews.hr_inbox"))

    return render_template("reviews/hr_review.html", review=review)
