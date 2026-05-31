from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints

app = FastAPI(
    title="Timestamp Reader API",
    description="API for extracting GPS coordinates and timestamp from photos using OCR",
    version="1.0.0"
)

# Configure CORS for SR Create frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include API router
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Timestamp Reader API. Check /docs for API documentation."}
