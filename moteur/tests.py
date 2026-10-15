import hashlib
from unittest.mock import Mock, patch

import numpy as np
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Document, EvalQuestion
from .services import embed, ollama_ok


@override_settings(
    OLLAMA_URL="http://ollama.example:11434",
    OLLAMA_API_KEY="test-token",
    OLLAMA_TIMEOUT=45,
    OLLAMA_HEALTH_TIMEOUT=7,
)
class RemoteOllamaTest(TestCase):
    @patch("moteur.services.requests.post")
    def test_embed_uses_remote_url_auth_and_configured_timeout(self, post):
        response = Mock()
        response.json.return_value = {"embeddings": [[3.0, 4.0]]}
        post.return_value = response

        result = embed(["document"])

        post.assert_called_once_with(
            "http://ollama.example:11434/api/embed",
            json={"model": "bge-m3", "input": ["document"]},
            headers={"Authorization": "Bearer test-token"},
            timeout=45,
        )
        np.testing.assert_allclose(result, [[0.6, 0.8]])

    @patch("moteur.services.requests.post")
    def test_embed_falls_back_to_legacy_endpoint_on_404(self, post):
        unsupported_endpoint = Mock(status_code=404)
        legacy_response = Mock()
        legacy_response.json.return_value = {"embedding": [3.0, 4.0]}
        post.side_effect = [unsupported_endpoint, legacy_response]

        result = embed(["document"])

        self.assertEqual(post.call_count, 2)
        _, kwargs = post.call_args_list[1]
        self.assertEqual(kwargs["json"], {"model": "bge-m3", "prompt": "document"})
        self.assertIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["timeout"], 45)
        self.assertEqual(
            post.call_args_list[1].args[0],
            "http://ollama.example:11434/api/embeddings",
        )
        np.testing.assert_allclose(result, [[0.6, 0.8]])

    @patch("moteur.services.requests.get")
    def test_health_check_uses_remote_url_auth_and_configured_timeout(self, get):
        get.return_value = Mock(ok=True)

        self.assertTrue(ollama_ok())

        get.assert_called_once_with(
            "http://ollama.example:11434/api/tags",
            headers={"Authorization": "Bearer test-token"},
            timeout=7,
        )


def fake_embed(texts):
    """Embedding factice (sac de mots haché) pour tester sans Ollama."""
    M = np.zeros((len(texts), 64), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            M[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return M / np.linalg.norm(M, axis=1, keepdims=True)


@patch("moteur.services.embed", fake_embed)
class PipelineTest(TestCase):
    def setUp(self):
        self.a = Document.objects.create(title="Gradient", text="descente de gradient optimisation réseau neurones")
        self.b = Document.objects.create(title="SQL", text="jointure requête base de données relationnelle")

    def test_pipeline(self):
        self.client.post(reverse("build"))
        self.assertEqual(Document.objects.exclude(embedding=None).count(), 2)
        r = self.client.get(reverse("search"), {"q": "requête base de données"})
        self.assertContains(r, "SQL")
        EvalQuestion.objects.create(question="gradient optimisation", expected=self.a)
        EvalQuestion.objects.create(question="jointure requête", expected=self.a)  # erreur voulue
        r = self.client.post(reverse("evaluation"), {"run": "1"})
        self.assertEqual(r.context["metrics"]["n"], 2)
        self.assertEqual(len(r.context["errors"]), 1)
        self.assertEqual(self.client.get(reverse("export")).status_code, 200)
