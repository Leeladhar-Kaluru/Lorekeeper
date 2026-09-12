from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.database.models import Message

import discord


class MessageRepository:
    """Handles database operations for Discord messages."""

    def __init__(self, session: Session):
        self.session = session

    def save_message(self, message: discord.Message) -> None:
        """Insert a Discord message or update it if it already exists."""

        statement = insert(Message).values(
            message_id=message.id,
            server_id=message.guild.id,
            server_name=message.guild.name,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            author_id=message.author.id,
            author_name=str(message.author),
            content=message.content,
            timestamp=message.created_at,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[Message.message_id],
            set_={
                "content": statement.excluded.content,
                "channel_name": statement.excluded.channel_name,
                "author_name": statement.excluded.author_name,
            },
        )

        self.session.execute(statement)
        self.session.commit()

    def save_messages(self, messages: list[discord.Message])->None:

        if not messages:
            return

        values = [
            {
                "message_id": message.id,
                "server_id": message.guild.id,
                "server_name": message.guild.name,
                "channel_id": message.channel.id,
                "channel_name": message.channel.name,
                "author_id": message.author.id,
                "author_name": message.author.display_name,
                "content": message.content,
                "timestamp": message.created_at,
            }
            for message in messages
        ]

        statement = insert(Message).values(values)

        statement = statement.on_conflict_do_update(
            index_elements=[Message.message_id],
            set_={
                "content": statement.excluded.content,
                "channel_name": statement.excluded.channel_name,
                "author_name": statement.excluded.author_name,
            },
        )

        self.session.execute(statement)
        self.session.commit()

    def exists(self, message_id: int) -> bool:
        """Check if a message exists."""

        statement = select(Message).where(Message.message_id == message_id)
        if self.session.scalar(statement) is not None:
            return True
        return False

    def get_message(self,message_id:int)->Message | None:

        statement = select(Message).where(Message.message_id==message_id)
        return self.session.scalar(statement)

    def get_nearby_messages(
        self,
        channel_id: int,
        timestamp: datetime,
        before: int = 10,
        after: int = 10,
    ) -> list[Message]:
        """Get messages surrounding a particular message."""

        before_statement = (
            select(Message)
            .where(
                Message.channel_id == channel_id,
                Message.timestamp < timestamp,
            )
            .order_by(Message.timestamp.desc())
            .limit(before)
        )

        after_statement = (
            select(Message)
            .where(
                Message.channel_id == channel_id,
                Message.timestamp > timestamp,
            )
            .order_by(Message.timestamp.asc())
            .limit(after)
        )

        before_messages = list(
            self.session.scalars(before_statement)
        )

        after_messages = list(
            self.session.scalars(after_statement)
        )

        before_messages.reverse()

        return before_messages + after_messages

    def get_recent_messages(
        self,
        channel_id: int,
        limit: int = 100,
    )->list[Message]:
        """Let's get the recent messages"""

        statement = (
            select(Message)
            .where(Message.channel_id==channel_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )

        messages = list(self.session.scalars(statement))

        messages.reverse()

        return messages

    def get_distinct_authors(self) -> list[str]:

        statement = (
            select(Message.author_name)
            .distinct()
            .order_by(Message.author_name)
        )

        return list(self.session.scalars(statement))

    def execute_query(self,sql_query: str):

        statement = text(sql_query)

        result = self.session.execute(statement).all()

        return [dict(row._mapping) for row in result]
         