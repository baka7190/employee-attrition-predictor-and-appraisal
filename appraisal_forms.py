# appraisal_forms.py
from flask_wtf import FlaskForm
from wtforms import (
    StringField, DateField, IntegerField, DecimalField, TextAreaField,
    SelectField, HiddenField, FieldList, FormField
)
from wtforms.validators import Optional, NumberRange, Length, DataRequired
from wtforms import Form  # for subforms inside FieldList


# ---------- Subforms ----------
class TeachRowForm(Form):
    unit_code = StringField('Unit code', validators=[Optional(), Length(max=50)])
    unit_name = StringField('Unit name', validators=[Optional(), Length(max=200)])
    group_name = StringField('Group', validators=[Optional(), Length(max=50)])
    num_students = IntegerField('No. of students', validators=[Optional(), NumberRange(min=0)])
    session_type = StringField('Session type', validators=[Optional(), Length(max=120)])
    hours_per_week = DecimalField('Hours / week', places=1, validators=[Optional(), NumberRange(min=0)])


class SkillRatingRow(Form):
    label_text = HiddenField()
    # supervisor fills later on a different screen
    rating = SelectField(
        'Rating',
        choices=[('', ''), ('1','1'), ('2','2'), ('3','3'), ('4','4'), ('5','5')]
    )


def _safe_int_coerce(v):
    """Coerce to int, but return None for blanks so employee POST won't explode."""
    s = '' if v is None else str(v).strip()
    return int(s) if s.isdigit() else None


class ScoreForm(Form):
    criteria_id = IntegerField("Criteria ID")
    # IMPORTANT: custom coerce + validate_choice=False to avoid 'Not a valid choice' on empty employee POST
    rating = SelectField(
        "Rating",
        choices=[(5, "Superior"), (4, "More than satisfactory"), (3, "Satisfactory"),
                 (2, "Needs Improvement"), (1, "Unsatisfactory")],
        coerce=_safe_int_coerce,
        validate_choice=False,
        default=3,
    )
    example_text = TextAreaField("Example", render_kw={"rows": 2})


# ---------- Main Form ----------
class AppraisalForm(FlaskForm):
    # A. Staff info (employee sees; name/position/department render read-only in template)
    employee_name   = StringField('Employee name')
    hod_name        = StringField('HOD name')
    position_title  = StringField('Position')
    department_name = StringField('Department')
    entry_date      = DateField('Entry date to DWU', validators=[Optional()])

    # Supervisor-only meta (locked in employee view)
    hod_position     = StringField('Head of Department (position)')
    supervise_years  = IntegerField('Length supervising (years)', validators=[Optional(), NumberRange(min=0)])
    supervise_months = IntegerField('Length supervising (months)', validators=[Optional(), NumberRange(min=0, max=11)])

    # B. Period
    period_start = DateField('Period Start', validators=[DataRequired()])
    period_end   = DateField('Period End', validators=[DataRequired()])

    # C. Time & Effort split
    teaching_percent = DecimalField('Teaching & learning %', places=1, validators=[Optional(), NumberRange(min=0, max=100)])
    research_percent = DecimalField('Research & publication %', places=1, validators=[Optional(), NumberRange(min=0, max=100)])
    other_percent    = DecimalField('Other commitments %', places=1, validators=[Optional(), NumberRange(min=0, max=100)])

    # D. Publications
    publications_2020_2024 = TextAreaField('Publications (2020–2024)', validators=[Optional(), Length(max=4000)])

    # Year token (for headings)
    eval_year = HiddenField()

    # E & F. Teaching workload
    teach_s2 = FieldList(FormField(TeachRowForm), min_entries=4, max_entries=20)
    teach_s1 = FieldList(FormField(TeachRowForm), min_entries=3, max_entries=20)

    # G. Duties/tasks (examples by employee; supervisor rates later)
    scores = FieldList(FormField(ScoreForm), min_entries=0, max_entries=50)

    # H. Personal skills (supervisor rates later)
    skill_ratings = FieldList(FormField(SkillRatingRow), min_entries=10, max_entries=20)

    # I & J. Comments + signatures
    employee_comment      = TextAreaField('Employee comments', validators=[Optional(), Length(max=4000)])
    employee_signature    = StringField("Employee's signature", validators=[Optional(), Length(max=120)])
    employee_sign_date    = DateField('Employee sign date', validators=[Optional()])
    supervisor_comments   = TextAreaField("Supervisor's comments", validators=[Optional(), Length(max=4000)])
    supervisor_signature  = StringField('Supervisor signature', validators=[Optional(), Length(max=120)])
    supervisor_sign_date  = DateField('Supervisor sign date', validators=[Optional()])
