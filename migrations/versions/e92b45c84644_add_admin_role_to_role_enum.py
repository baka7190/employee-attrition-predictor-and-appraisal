from alembic import op

# revision identifiers, used by Alembic.
revision = 'xxxx'   # auto-generated filename id
down_revision = '9d9bd3350e56'  # keep what Alembic generated
branch_labels = None
depends_on = None

def upgrade():
    # Add ADMIN to role_enum in PostgreSQL
    op.execute("ALTER TYPE role_enum ADD VALUE IF NOT EXISTS 'ADMIN';")

def downgrade():
    # PostgreSQL does not support removing enum values directly.
    # To rollback, you'd need to recreate the type without ADMIN.
    pass
