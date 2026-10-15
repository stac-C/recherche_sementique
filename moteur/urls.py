from django.urls import path
from . import views

urlpatterns = [
    path("", views.documents, name="documents"),
    path("documents/<int:pk>/supprimer/", views.delete_document, name="delete_document"),
    path("index/", views.build, name="build"),
    path("recherche/", views.search, name="search"),
    path("evaluation/", views.evaluation, name="evaluation"),
    path("evaluation/question/<int:pk>/supprimer/", views.delete_question, name="delete_question"),
    path("evaluation/export/", views.export, name="export"),
]
