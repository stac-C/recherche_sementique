import csv
import io
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import services
from .models import Document, EvalQuestion


def documents(request):
    """Étape 1 : importer et gérer le corpus."""
    if request.method == "POST":
        if "delete_selected" in request.POST:
            selected_ids = request.POST.getlist("selected_documents")
            if not selected_ids:
                messages.warning(request, "Sélectionnez au moins un document à supprimer.")
                return redirect("documents")

            selected = Document.objects.filter(pk__in=selected_ids)
            count = selected.count()
            question_count = EvalQuestion.objects.filter(expected__in=selected).count()
            selected.delete()
            messages.success(
                request,
                f"{count} document(s) supprimé(s) et {question_count} question(s) "
                "de test associée(s).",
            )
            return redirect("documents")

        n = 0
        for f in request.FILES.getlist("files"):
            raw = f.read().decode("utf-8-sig", "ignore")
            if f.name.lower().endswith(".csv"):
                for row in csv.DictReader(io.StringIO(raw)):
                    if (row.get("text") or "").strip():
                        Document.objects.create(title=row.get("title") or "sans titre", text=row["text"].strip())
                        n += 1
            elif raw.strip():
                Document.objects.create(title=Path(f.name).stem, text=raw.strip())
                n += 1
        messages.success(request, f"{n} document(s) importé(s). Pense à régénérer l'index.")
        return redirect("documents")
    return render(request, "moteur/documents.html", {"docs": Document.objects.all()})


@require_POST
def delete_document(request, pk):
    get_object_or_404(Document, pk=pk).delete()
    return redirect("documents")


def build(request):
    """Étape 2 : embeddings."""
    if request.method == "POST":
        try:
            r = services.build_index()
            messages.success(request, f"{r['n']} vecteurs de dimension {r['dim']} en {r['seconds']} s.")
        except Exception as e:
            messages.error(request, f"Échec : {e}")
        return redirect("build")
    total = Document.objects.count()
    indexed = Document.objects.exclude(embedding=None).count()
    return render(request, "moteur/build.html", {
        "total": total, "indexed": indexed, "ollama": services.ollama_ok(),
        "model": settings.EMBED_MODEL, "url": settings.OLLAMA_URL})


def search(request):
    """Étape 3 : recherche sémantique."""
    q = request.GET.get("q", "").strip()
    k = int(request.GET.get("k", 5))
    total = Document.objects.count()
    indexed = Document.objects.exclude(embedding=None).count()
    results, error = [], None
    if q:
        try:
            results = services.search(q, k)
            if not results:
                error = "Aucun document indexé."
        except Exception as e:
            error = str(e)
    return render(request, "moteur/search.html", {
        "q": q, "k": k, "results": results, "error": error,
        "total": total, "indexed": indexed,
    })


def evaluation(request):
    """Étape 4 : jeu de test + analyse des erreurs."""
    if request.method == "POST" and "question" in request.POST:
        q = request.POST["question"].strip()
        if q and request.POST.get("expected"):
            EvalQuestion.objects.create(question=q, expected_id=request.POST["expected"])
        return redirect("evaluation")
    docs = Document.objects.all()
    total = docs.count()
    indexed = docs.exclude(embedding=None).count()
    questions = EvalQuestion.objects.select_related("expected")
    ctx = {
        "docs": docs,
        "questions": questions,
        "total": total,
        "indexed": indexed,
        "can_evaluate": bool(total and indexed == total and questions.exists()),
    }
    if request.method == "POST" and "run" in request.POST:
        try:
            rows, metrics = services.evaluate()
            ctx.update(rows=rows, metrics=metrics, errors=[r for r in rows if r["rank"] > 1])
        except Exception as e:
            messages.error(request, str(e))
    return render(request, "moteur/evaluation.html", ctx)


@require_POST
def delete_question(request, pk):
    get_object_or_404(EvalQuestion, pk=pk).delete()
    return redirect("evaluation")


def export(request):
    rows, _ = services.evaluate()
    resp = HttpResponse(content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = 'attachment; filename="resultats.csv"'
    w = csv.writer(resp)
    w.writerow(["question", "attendu", "top1", "rang_attendu", "score_top1", "score_attendu", "diagnostic"])
    for r in rows:
        w.writerow([r["question"], r["expected"].title, r["top1"].title, r["rank"], r["s_top"], r["s_exp"], r["diag"]])
    return resp
