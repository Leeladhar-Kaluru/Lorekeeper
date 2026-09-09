import discord

class HistoricalMessageLoader:

    async def load(self,channel:discord.TextChannel,limit:int=100)->list[discord.Message]:

        "let's fetch the most recent messages from the text channel"
        messages = []

        async for message in channel.history(limit=limit):
            if message.author.bot:
                continue
            messages.append(message)

        messages.reverse()

        return messages
        