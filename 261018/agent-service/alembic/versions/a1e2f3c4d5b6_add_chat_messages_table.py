"""add chat messages table

Revision ID: a1e2f3c4d5b6
Revises: c5adae5bb194
Create Date: 2026-10-22 16:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1e2f3c4d5b6'
down_revision: Union[str, Sequence[str], None] = 'c5adae5bb194'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('chat_messages',
    sa.Column('message_id', sa.String(length=64), nullable=False),
    sa.Column('thread_id', sa.String(length=64), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('user_text', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('final_answer', sa.Text(), nullable=True),
    sa.Column('sources', sa.JSON(), nullable=True),
    sa.Column('intent', sa.String(length=32), nullable=True),
    sa.Column('needs_human', sa.String(length=1), nullable=True),
    sa.Column('degraded', sa.String(length=1), nullable=True),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('chunk_count', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('message_id')
    )
    op.create_index('ix_chat_messages_thread', 'chat_messages', ['thread_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_chat_messages_thread', table_name='chat_messages')
    op.drop_table('chat_messages')
