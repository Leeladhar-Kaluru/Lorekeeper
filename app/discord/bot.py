import os

import discord
from discord import app_commands
from dotenv import load_dotenv
from langchain_core.documents import Document

from app.discord.message_converter import DiscordMessageConverter
from app.discord.historical_loader import HistoricalMessageLoader

from app.rag.splitter import DocumentSplitter
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.rag_service import RAGService

from app.llm.groq_service import GroqService

load_dotenv()

class LoreKeeper(discord.Client):

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(intents=intents)

        self.splitter = DocumentSplitter()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            embedding_function = self.embedding_service,
            collection_name="discord-lore"
        )
        self.rag_service = RAGService(
            vector_store=self.vector_store,
            llm_service=GroqService()
        )

        self.tree = app_commands.CommandTree(self)


    def ingest_documents(self,documents:list[Document])->None:
        """ingest documents into vector store"""

        if not documents:
            raise ValueError("Documents cannot be empty")
        
        print(f"ingesting {len(documents)} documents")

        chunks = self.splitter.split(documents)

        print(f"Created {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)

        print("Stored chunks in vector store")

    async def on_ready(self):
        print(f"Logged in as {self.user}")

        synced = await self.tree.sync()

        print(f"Synced {len(synced)} commands")

        for guild in self.guilds:
            print(f"\nServer: {guild.name}")

            for channel in guild.text_channels:
                print(f"Channel: #{channel.name}")

                loader = HistoricalMessageLoader()

                messages = await loader.load(
                    channel=channel,
                    limit=100,
                )

                documents = [DiscordMessageConverter.convert(message) for message in messages]

                print(f"Loaded {len(documents)} documents")

                self.ingest_documents(documents)

    async def on_message(self, message: discord.Message):

        if message.author.bot:
            return

        document = DiscordMessageConverter.convert(message)

        print("\n--- Discord Document ---")
        print(f"Content: {document.page_content}")
        print(f"Metadata: {document.metadata}")

        self.ingest_documents([document])

        

def run_bot():

    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        raise ValueError("Token is not set")

    bot = LoreKeeper()

    @bot.tree.command(
        name="ask",
        description="Ask Lorekeeper about the server's memories"
    )

    async def ask(interaction: discord.Interaction, question: str):

        answer = bot.rag_service.answer(question)
        await interaction.response.send_message(answer)

    bot.run(token)
