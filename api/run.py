"""Start the Kronos Prediction API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9188,
        reload=False,
        workers=1,
    )
