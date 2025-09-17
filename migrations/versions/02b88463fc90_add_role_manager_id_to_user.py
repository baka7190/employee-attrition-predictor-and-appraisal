"""Add role + manager_id to user

Revision ID: 02b88463fc90
Revises: 
Create Date: 2025-08-21 10:48:20.360477
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '02b88463fc90'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()

    # 1) Create the PostgreSQL ENUM type first
    role_enum = postgresql.ENUM('EMPLOYEE', 'MANAGER', 'HR', name='role_enum')
    role_enum.create(bind, checkfirst=True)

    # 2) Ensure a 'role' column exists as VARCHAR to convert from
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS role VARCHAR(20)')

    # 3) Drop any default before type change to avoid cast errors on defaults
    op.execute('ALTER TABLE "user" ALTER COLUMN role DROP DEFAULT')

    # 4) Normalize existing values to valid enum labels (case-insensitive)
    # Map common strings to our canonical enum values
    op.execute("""UPDATE "user" SET role = 'HR'
                  WHERE role IS NOT NULL AND lower(role) IN ('admin','hr','human resources','human_resources')""")
    op.execute("""UPDATE "user" SET role = 'MANAGER'
                  WHERE role IS NOT NULL AND lower(role) IN ('manager','mgr','hod','head of department','head_of_department')""")
    op.execute("""UPDATE "user" SET role = 'EMPLOYEE'
                  WHERE role IS NOT NULL AND lower(role) IN ('employee','user','staff','worker')""")

    # Backfill anything null/empty or still invalid to EMPLOYEE
    op.execute("""UPDATE "user" SET role = 'EMPLOYEE' WHERE role IS NULL OR role = '' """)
    op.execute("""UPDATE "user" SET role = 'EMPLOYEE'
                  WHERE role NOT IN ('EMPLOYEE','MANAGER','HR')""")

    # 5) Convert to ENUM and make NOT NULL
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column(
            'role',
            existing_type=sa.VARCHAR(length=20),
            type_=role_enum,
            nullable=False,
            postgresql_using='role::text::role_enum'
        )
        # 6) Add manager_id and FK
        batch_op.add_column(sa.Column('manager_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_user_manager', 'user', ['manager_id'], ['id'])


def downgrade():
    bind = op.get_bind()
    role_enum = postgresql.ENUM('EMPLOYEE', 'MANAGER', 'HR', name='role_enum')

    with op.batch_alter_table('user', schema=None) as batch_op:
        # Drop FK and column
        try:
            batch_op.drop_constraint('fk_user_manager', type_='foreignkey')
        except Exception:
            pass
        try:
            batch_op.drop_column('manager_id')
        except Exception:
            pass

        # Convert ENUM back to VARCHAR (allow NULLs again)
        batch_op.alter_column(
            'role',
            existing_type=role_enum,
            type_=sa.VARCHAR(length=20),
            nullable=True,
            postgresql_using='role::text'
        )

    # Finally drop the ENUM type
    role_enum.drop(bind, checkfirst=True)
