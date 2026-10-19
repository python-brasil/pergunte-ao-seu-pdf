# Pergunte ao seu PDF

Busca semântica em qualquer PDF, rodando 100% na sua máquina. É a metade
"retrieval" de um RAG, em 40 linhas de Python.

```
pip install pypdf sentence-transformers numpy
python main.py contrato.pdf
```

Como funciona:

1. `pypdf` extrai o texto de cada página.
2. O texto é cortado em pedaços de 500 caracteres, com 100 de sobreposição.
3. Cada pedaço vira um vetor com o modelo `paraphrase-multilingual-MiniLM-L12-v2`.
4. A pergunta vira vetor também, e os 3 pedaços mais parecidos (cosseno) voltam.

Para virar um RAG completo, mande os trechos devolvidos junto com a pergunta
para o modelo de linguagem da sua escolha.

Publicado pelo [@python_brasil](https://instagram.com/python_brasil).
