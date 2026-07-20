import importlib
import json
import os
import sys
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_embedding_service_uses_bedrock_when_configured(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    module = importlib.import_module("services.embeddings")
    importlib.reload(module)

    class FakeBedrockClient:
        def invoke_model(self, modelId, body, contentType, accept):
            assert modelId == "amazon.titan-embed-text-v2:0"
            payload = json.loads(body)
            assert payload["inputText"] == "hello"
            return {
                "body": type("Body", (), {"read": lambda self: json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode("utf-8")})()
            }

    with patch("services.embeddings.boto3.client", return_value=FakeBedrockClient()):
        service = module.EmbeddingService()
        assert service.embed_query("hello") == [0.1, 0.2, 0.3]
