from fastapi.testclient import TestClient
from src.api import create_app


class Engine:
    def generate(self, context, question, max_new_tokens):
        if not question.strip():
            raise ValueError("Question must be nonempty")
        return {"answer": "base weights", "prompt_tokens": 10, "generated_tokens": 2, "latency_seconds": 0.01}


def test_auth_health_and_generation():
    with TestClient(create_app(engine=Engine(), api_key="unit-key")) as client:
        assert client.get("/health").json()["status"] == "ready"
        body = {"context": "LoRA freezes base weights.", "question": "What is frozen?"}
        assert client.post("/generate", json=body).status_code == 401
        response = client.post("/generate", json=body, headers={"Authorization": "Bearer unit-key"})
        assert response.status_code == 200 and response.json()["answer"] == "base weights"


def test_request_limits_and_invalid_prompts():
    with TestClient(create_app(engine=Engine())) as client:
        assert (
            client.post(
                "/generate", json={"context": "x", "question": "q", "max_new_tokens": 10000}
            ).status_code
            == 422
        )
        assert client.post("/generate", json={"context": "x" * 100001, "question": "q"}).status_code == 422
        assert client.post("/generate", json={"context": "x", "question": " "}).status_code == 422
