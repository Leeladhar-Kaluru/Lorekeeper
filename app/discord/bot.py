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

from app.llm.answer_generator import AnswerGenerator

from app.query.query_engine import QueryEngine

from app.database.connection import SessionLocal
from app.database.message_repository import MessageRepository

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
            vector_store=self.vector_store
        )

        self.query_engine = QueryEngine()

        self.answer_generator = AnswerGenerator()

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

        session = SessionLocal()

        try:

            message_repository = MessageRepository(session)

            for guild in self.guilds:
                print(f"\nServer: {guild.name}")

                for channel in guild.text_channels:
                    print(f"Channel: #{channel.name}")

                    loader = HistoricalMessageLoader()

                    messages = await loader.load(
                        channel=channel,
                        limit=100,
                    )
                    
                    message_repository.save_messages(messages)
            
                    documents = [DiscordMessageConverter.convert(message) for message in messages]

                    print(f"Loaded {len(documents)} documents")

                    self.ingest_documents(documents)

        finally:
            session.close()

    async def on_message(self, message: discord.Message):

        if message.author.bot:
            return

        session = SessionLocal()

        try:
            message_repository = MessageRepository(session)
            message_repository.save_message(message)

        finally:
            session.close()

        document = DiscordMessageConverter.convert(message)

        print("\n--- Discord Document ---")
        print(f"Content: {document.page_content}")
        print(f"Metadata: {document.metadata}")

        self.ingest_documents([document])


    @staticmethod
    async def send_long_message(interaction: discord.Interaction, content: str):
        
        for i in range(0, len(content), 2000):
            await interaction.followup.send(content[i:i + 2000])

        

def run_bot():

    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        raise ValueError("Token is not set")

    bot = LoreKeeper()

    @bot.tree.command(
    name="ask",
    description="Ask Lorekeeper about the server's memories"
    )
    async def ask(
        interaction: discord.Interaction,
        question: str,
    ):

        await interaction.response.defer()

        # ---------------------------------------------------------
        # 1. Generate SQL from the user's question
        # ---------------------------------------------------------

        sql_query = bot.query_engine.query_generator(question)

        # ---------------------------------------------------------
        # 2. Execute SQL
        # ---------------------------------------------------------

        session = SessionLocal()

        try:

            repository = MessageRepository(session)

            database_results = repository.execute_query(
                sql_query
            )

        finally:
            session.close()

        print("\n========== DATABASE RESULTS ==========")
        print(database_results)

        # ---------------------------------------------------------
        # 3. Semantic retrieval and Expanded Context
        # ---------------------------------------------------------

        semantic_context = bot.rag_service.retrieve(
            query=question,
            k=10,
        )

        # ---------------------------------------------------------
        # 4. Build final evidence
        # ---------------------------------------------------------

        final_context = f"""
        DATABASE RESULTS
        ================

        {database_results}


        SEMANTIC RETRIEVAL
        ==================

        {semantic_context}
        """

        # ---------------------------------------------------------
        # 5. Generate final answer
        # ---------------------------------------------------------

        answer = bot.answer_generator.generate(
            question=question,
            context=final_context,
        )

        response = (
            f"**Question**: {question}\n\n"
            f"**Answer**: {answer}"
        )

        await LoreKeeper.send_long_message(
            interaction,
            response,
        )

    bot.run(token)