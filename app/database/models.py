from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    server_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    server_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    channel_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    channel_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    author_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    author_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )