# face_verification_api/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('face-verification/', include('face_verification.urls')),
]