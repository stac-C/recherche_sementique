from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=200)
    text = models.TextField()
    embedding = models.BinaryField(null=True, blank=True)  # float32 normalisé
    embed_model = models.CharField(max_length=100, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.title


class EvalQuestion(models.Model):
    question = models.CharField(max_length=500)
    expected = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="questions")

    class Meta:
        ordering = ["id"]
