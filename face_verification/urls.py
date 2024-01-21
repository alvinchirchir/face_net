# face_verification/urls.py
from django.urls import path
from .views import face_verification_api

urlpatterns = [
    path('api/face-verification/', face_verification_api, name='face_verification_api'),
]
