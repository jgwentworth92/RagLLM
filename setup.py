from setuptools import setup, find_packages

setup(
    name="RAG LLM conversational Chatbot",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'pydantic',
        'langchain==0.1.12',
        'langchain-community==0.0.28',
        'langchain-core==0.1.32',
        'langchain-openai',
        'SQLAlchemy',
        'psycopg2-binary',
        'pgvector',
        'tiktoken',
        'semantic-text-splitter==0.7.0',
        'tokenizers',
        'asyncpg',
        'alembic',
        'pydantic',
        'pydantic-core',
        'pydantic-settings',
        "scikit-learn",
        'numpy~=1.26.4',
        'pandas~=2.2.1',
        'umap-learn==0.5.5',
        'anthropic',
        'icecream'
    ],
    entry_points={
        'my_fastapi_app.plugins': [
            'RagLLM = RagLLM.Router.routes:add_routes',
        ],
    }
)
