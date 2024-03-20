from setuptools import setup, find_packages

setup(
    name="RAG LLM conversational Chatbot",
    version="0.16",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'pydantic',
        'langchain',
        'langchain-community',
        'langchain-core',
        'langchain-openai',
        'SQLAlchemy',
        'psycopg2-binary',
        'pgvector',
        'tiktoken',
        'semantic-text-splitter==0.7.0',
        'pyautogen[redis]',
        'tokenizers',
        'asyncpg',
        'alembic',
        'pydantic',
        'pydantic-core',
        'pydantic-settings',
        "pyautogen[retrievechat]"
    ],
    entry_points={
        'my_fastapi_app.plugins': [
            'RagLLM = RagLLM.Router.routes:add_routes',
        ],
    }
)
