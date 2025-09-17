from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, HiddenField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    identifier = StringField("Identifier", validators=[DataRequired()])  # username OR email
    password = PasswordField("Password", validators=[DataRequired()])
    expected_role = HiddenField("Expected Role")  # optional hidden role
    submit = SubmitField("Sign In")
