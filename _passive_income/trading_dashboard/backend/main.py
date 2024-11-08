# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import auth, stocks, models

app = FastAPI(title="AI-Powered Trading Dashboard")

# CORS settings (adjust origins as needed)
origins = [
    "http://localhost",
    "http://localhost:3000",  # React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
app.include_router(models.router, prefix="/models", tags=["Models"])

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Trading Dashboard API"}
