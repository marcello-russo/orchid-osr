# app/api/main.py
from fastapi import FastAPI
from app.api.routes import health, classify

# Create FastAPI app
app = FastAPI(
    title="Orchid Classification API",
    description="API per classificare specie di orchidee Ophrys con ResNet",
    version="1.0.0",
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(classify.router, tags=["classification"])
