from langchain_core.documents import Document

import discord

class DiscordMessageConverter:
    """Let's convert discord messages into langchian Documents"""

    @staticmethod
    def convert(message: discord.Message)->Document:
        return Document(
            page_content=message.content,
            metadata={
                "source": "discord",
                "server_id": str(message.guild.id),
                "server_name": message.guild.name,
                "channel_id": str(message.channel.id),
                "channel_name": message.channel.name,
                "author_id": str(message.author.id),
                "author_name": str(message.author),
                "message_id": str(message.id),
                "timestamp": message.created_at.isoformat()
            }
        )