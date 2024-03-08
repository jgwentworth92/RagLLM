from setuptools import setup, find_packages

setup(
    name="RAG LLM conversational Chatbot",
    version="0.16",
    packages=find_packages(include=["app"]),
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
        'semantic-text-splitter',
        'tokenizers'
    ],
    entry_points={
        'my_fastapi_app.plugins': [
            'RagLLM = app.Router.routes:add_routes',
        ],
    }
)
