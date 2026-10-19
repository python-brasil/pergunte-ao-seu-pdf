"""Pergunte qualquer coisa a um PDF, sem API paga.

Le o PDF, corta em pedacos, transforma cada pedaco em vetor (embedding) com um
modelo que roda na sua maquina e, para cada pergunta, devolve os trechos mais
parecidos. E o coracao de um RAG em 40 linhas.

    pip install pypdf sentence-transformers numpy
    python main.py contrato.pdf
"""
import sys

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Modelo pequeno (118 MB), multilingue, baixa uma vez e fica em cache.
MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


def ler_pdf(caminho):
    texto = ""
    for pagina in PdfReader(caminho).pages:
        texto += (pagina.extract_text() or "") + "\n"
    return texto


def cortar(texto, tamanho=500, sobra=100):
    """Pedacos de ~500 caracteres com 100 de sobreposicao,
    para uma frase cortada no meio aparecer inteira no vizinho."""
    pedacos = []
    for i in range(0, len(texto), tamanho - sobra):
        pedacos.append(texto[i:i + tamanho])
    return pedacos


def buscar(pergunta, pedacos, vetores, modelo, k=3):
    v = modelo.encode([pergunta], normalize_embeddings=True)[0]
    notas = vetores @ v            # cosseno, porque tudo esta normalizado
    melhores = np.argsort(notas)[::-1][:k]
    return [(float(notas[i]), pedacos[i]) for i in melhores]


def main():
    if len(sys.argv) < 2:
        sys.exit("uso: python main.py arquivo.pdf")

    modelo = SentenceTransformer(MODELO)
    pedacos = cortar(ler_pdf(sys.argv[1]))
    vetores = modelo.encode(pedacos, normalize_embeddings=True)
    print(f"{len(pedacos)} trechos indexados. Pergunte (Ctrl+C sai).\n")

    while True:
        pergunta = input("> ")
        for nota, trecho in buscar(pergunta, pedacos, vetores, modelo):
            print(f"\n[{nota:.2f}] {trecho.strip()[:300]}...")
        print()


if __name__ == "__main__":
    main()
