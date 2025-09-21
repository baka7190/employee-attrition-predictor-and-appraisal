from datetime import datetime, date
from enum import Enum

from flask_login import UserMixin
from extensions import db


# --- Enums ---
class Role(Enum):
    EMPLOYEE = "EMPLOYEE"
    MANAGER = "MANAGER"
    HR = "HR"
    ADMIN = "ADMIN"


class ReviewStatus(Enum):
    DRAFT = "DRAFT"
    SUBMITTED_TO_MANAGER = "SUBMITTED_TO_MANAGER"
    MANAGER_SUBMITTED = "MANAGER_SUBMITTED"
    HR_REVIEWED = "HR_REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# --- User model ---
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    # ✅ Profile fields
    full_name    = db.Column(db.String(120))
    employee_id  = db.Column(db.String(50), unique=True)
    department   = db.Column(db.String(120))
    job_title    = db.Column(db.String(120))
    manager_name = db.Column(db.String(120))
    phone        = db.Column(db.String(50))
    location     = db.Column(db.String(120))

    role = db.Column(
        db.Enum(Role, name="role_enum"),
        nullable=False,
        default=Role.EMPLOYEE,
        server_default="EMPLOYEE",
    )

    is_active = db.Column(db.Boolean, default=False)

    # self-referential manager
    manager_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    manager = db.relationship("User", remote_side=[id])

    # ---- Helpful relationship collections (enable cascades) ----
    # predictions owned by this user (delete with user)
    predictions = db.relationship(
        "Prediction",
        backref=db.backref("user", lazy=True),
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )

    # appraisals where this user is the employee (delete with user)
    employee_appraisals = db.relationship(
        "Appraisal",
        foreign_keys="Appraisal.employee_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )

    # appraisals where this user is the supervisor (keep appraisals; SET NULL)
    supervisor_appraisals = db.relationship(
        "Appraisal",
        foreign_keys="Appraisal.supervisor_id",
        passive_deletes=True,
        lazy=True,
    )

    # performance review collections already work via backrefs below:
    #  - employee_reviews (CASCADE) from PerformanceReview.employee
    #  - manager_reviews / hr_reviews (SET NULL) from PerformanceReview.manager/hr

    # retention plans (delete with user)
    retention_plans = db.relationship(
        "RetentionPlan",
        backref=db.backref("user", lazy=True),
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )

    @property
    def name(self):
        return self.full_name or self.username

    @name.setter
    def name(self, value):
        self.full_name = value

    def __repr__(self):
        return f"<User {self.username} ({self.email}) Active={self.is_active}>"


# --- Prediction model ---
class Prediction(db.Model):
    __tablename__ = "predictions"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    input_data = db.Column(db.JSON, nullable=True)
    result     = db.Column(db.String(50))
    confidence = db.Column(db.String(50))
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


# --- Appraisal Cycle models ---
class AppraisalCycle(db.Model):
    __tablename__ = "appraisal_cycle"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(255), nullable=False, unique=True)
    start_date = db.Column(db.Date, nullable=False)
    end_date   = db.Column(db.Date, nullable=False)
    status     = db.Column(db.String(20), default="Draft")  # Draft | Open | Closed
    template   = db.Column(db.JSON)

    enrollments = db.relationship(
        "CycleEnrollment",
        backref="cycle",
        lazy=True,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CycleEnrollment(db.Model):
    __tablename__ = "cycle_enrollment"

    id          = db.Column(db.Integer, primary_key=True)
    cycle_id    = db.Column(
        db.Integer,
        db.ForeignKey("appraisal_cycle.id", ondelete="CASCADE"),
        nullable=False
    )
    employee_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    manager_id  = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    status      = db.Column(db.String(30), default="Draft")

    employee = db.relationship("User", foreign_keys=[employee_id])
    manager  = db.relationship("User", foreign_keys=[manager_id])


# --- Performance Review ---
class PerformanceReview(db.Model):
    __tablename__ = "performance_reviews"

    id = db.Column(db.Integer, primary_key=True)

    period_start = db.Column(db.Date, nullable=False)
    period_end   = db.Column(db.Date, nullable=False)

    employee_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    manager_id  = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    hr_id       = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    # backrefs create user.employee_reviews / user.manager_reviews / user.hr_reviews
    employee = db.relationship("User", foreign_keys=[employee_id], backref="employee_reviews")
    manager  = db.relationship("User", foreign_keys=[manager_id],  backref="manager_reviews")
    hr       = db.relationship("User", foreign_keys=[hr_id],       backref="hr_reviews")

    self_goals        = db.Column(db.Text)
    self_achievements = db.Column(db.Text)
    self_training     = db.Column(db.Text)
    self_rating       = db.Column(db.Integer)

    manager_comments = db.Column(db.Text)
    manager_rating   = db.Column(db.Integer)

    hr_comments           = db.Column(db.Text)
    promotion_recommended = db.Column(db.Boolean, default=False)
    new_title             = db.Column(db.String(120))
    new_salary            = db.Column(db.Float)
    effective_date        = db.Column(db.Date)

    status = db.Column(
        db.Enum(ReviewStatus, name="review_status_enum"),
        nullable=False,
        default=ReviewStatus.DRAFT,
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    audits = db.relationship(
        "ReviewAudit",
        backref="review",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )


# --- Criteria-based Appraisal ---
class Appraisal(db.Model):
    __tablename__ = "appraisals"

    publications_text = db.Column(db.Text, nullable=True)
    teach_s1_json     = db.Column(db.JSON, nullable=True)
    teach_s2_json     = db.Column(db.JSON, nullable=True)

    id = db.Column(db.Integer, primary_key=True)

    employee_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    supervisor_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    # explicit relationships; we keep names you used elsewhere
    employee_user   = db.relationship("User", foreign_keys=[employee_id], passive_deletes=True)
    supervisor_user = db.relationship("User", foreign_keys=[supervisor_id], passive_deletes=True)

    period_start = db.Column(db.Date, nullable=False)
    period_end   = db.Column(db.Date, nullable=False)

    total_score        = db.Column(db.Integer, default=0)
    employee_comment   = db.Column(db.Text)
    supervisor_comment = db.Column(db.Text)

    status = db.Column(
        db.Enum(ReviewStatus, name="review_status_enum"),
        nullable=False,
        default=ReviewStatus.DRAFT,
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    scores = db.relationship(
        "Score",
        backref="appraisal",
        lazy=True,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    workload = db.relationship(
        "Workload",
        backref="appraisal",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Criteria(db.Model):
    __tablename__ = "criteria"

    id   = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50))  # "Duty" or "Skill"


class ReviewAudit(db.Model):
    __tablename__ = "review_audits"

    id         = db.Column(db.Integer, primary_key=True)
    review_id  = db.Column(
        db.Integer,
        db.ForeignKey("performance_reviews.id", ondelete="CASCADE"),
        nullable=False
    )
    actor_id   = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    from_status = db.Column(db.Enum(ReviewStatus, name="review_status_enum"), nullable=True)
    to_status   = db.Column(db.Enum(ReviewStatus, name="review_status_enum"), nullable=True)
    note       = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Score(db.Model):
    __tablename__ = "scores"

    id           = db.Column(db.Integer, primary_key=True)
    appraisal_id = db.Column(
        db.Integer,
        db.ForeignKey("appraisals.id", ondelete="CASCADE"),
        nullable=False
    )
    criteria_id  = db.Column(
        db.Integer,
        db.ForeignKey("criteria.id"),
        nullable=False
    )
    rating       = db.Column(db.Integer, nullable=False)
    example_text = db.Column(db.Text)


class Workload(db.Model):
    __tablename__ = "workloads"

    id           = db.Column(db.Integer, primary_key=True)
    appraisal_id = db.Column(
        db.Integer,
        db.ForeignKey("appraisals.id", ondelete="CASCADE"),
        nullable=False
    )
    teaching_percent = db.Column(db.Integer, default=0)
    research_percent = db.Column(db.Integer, default=0)
    other_percent    = db.Column(db.Integer, default=0)


# --- NEW RetentionPlan ---
class RetentionPlan(db.Model):
    __tablename__ = "retention_plans"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    immediate_plan = db.Column(db.Text, nullable=False)
    next_steps = db.Column(db.Text)
    assigned_to = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
