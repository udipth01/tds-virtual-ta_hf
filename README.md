---
title: TDS Virtual TA FAPI
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: docker
app_file: DockerFile
---
# TDS Virtual TA

An API for answering IITM Tools in Data Science (TDS) questions using FAISS and sentence-transformers.

## How to Run

```bash
uvicorn main:app --reload
