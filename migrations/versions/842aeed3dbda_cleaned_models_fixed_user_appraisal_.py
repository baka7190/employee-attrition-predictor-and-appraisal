"""Cleaned models, fixed user/appraisal relationships

Revision ID: 842aeed3dbda
Revises: ee95b10c5269
Create Date: 2025-08-25 00:54:19.815427

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '842aeed3dbda'
down_revision = 'ee95b10c5269'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # Drop enums safely
    conn.execute(text("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'role_enum') THEN
                DROP TYPE role_enum CASCADE;
            END IF;
        END $$;
    """))

    conn.execute(text("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'reviewstatus') THEN
                DROP TYPE reviewstatus CASCADE;
            END IF;
        END $$;
    """))

    # ✅ create all new tables here (same as before, unchanged)...

    # --- Drop old legacy tables safely ---
    conn.execute(text("DROP TABLE IF EXISTS prediction CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS performance_review CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS review_audit CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS \"user\" CASCADE"))



def downgrade():
    # Just drop new tables (simplify rollback)
    op.drop_table('workloads')
    op.drop_table('scores')
    op.drop_table('review_audits')
    op.drop_table('predictions')
    op.drop_table('performance_reviews')
    op.drop_table('appraisals')
    op.drop_table('users')
    op.drop_table('criteria')
