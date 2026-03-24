#!/bin/bash
cd ~/stable-money-avatar/stable-money-avatar
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
